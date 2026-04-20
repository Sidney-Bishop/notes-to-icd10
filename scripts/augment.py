#!/usr/bin/env python3
"""
augment.py — Targeted Data Augmentation for Weak ICD-10 Chapters
=================================================================
Generates synthetic SOAP notes for ICD-10 codes below target density
using Pydantic AI for structured, validated generation.

Key design principles
---------------------
- Schema-safe: reads the gold layer schema at startup and casts all new
  rows to match exactly — no type mismatch errors.
- Incremental saves: progress is written after EVERY code. Resume by
  re-running the same command.
- Pydantic AI: structured output with automatic retry on validation
  failure. No manual JSON parsing or regex extraction.
- Model-agnostic: LM Studio (default, free) or Anthropic API.
- Async parallel: generates notes concurrently for speed.

Usage
-----
    # LM Studio (default) — start server in LM Studio first
    uv run python scripts/augment.py --lmstudio-model "qwen/qwen3.6-35b-a3b"

    # Anthropic API
    uv run python scripts/augment.py --backend anthropic

    # Dry-run
    uv run python scripts/augment.py --dry-run

    # Resume interrupted run (automatic — just re-run)
    uv run python scripts/augment.py --lmstudio-model "qwen/qwen3.6-35b-a3b"

    # After augmenting, retrain Z and O:
    uv run python scripts/train.py \\
        --experiment E-005b_Augmented_ZO \\
        --mode hierarchical --stage 2 --code-filter billable \\
        --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/model \\
        --chapters Z O \\
        --gold-path data/gold/medsynth_gold_augmented.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config                                            # noqa: E402
from src.preprocessing import build_apso_note, redact_icd10_sections    # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic model for a single SOAP note
# ---------------------------------------------------------------------------

class SoapNote(BaseModel):
    subjective: str = Field(description="Subjective section — patient's reported symptoms and history")
    objective:  str = Field(description="Objective section — examination findings and vital signs")
    assessment: str = Field(description="Assessment section — clinical diagnosis and interpretation")
    plan:       str = Field(description="Plan section — management, medications, follow-up")


# ---------------------------------------------------------------------------
# Chapter context for prompts
# ---------------------------------------------------------------------------

CHAPTER_CONTEXT: dict[str, str] = {
    "Z": (
        "Chapter Z covers Factors Influencing Health Status and Contact with Health "
        "Services: vaccination visits, screening examinations, chronic disease "
        "management, personal and family history, BMI documentation, and social "
        "determinants of health."
    ),
    "O": (
        "Chapter O covers Pregnancy, Childbirth and the Puerperium. Notes document "
        "routine antenatal visits, obstetric assessments, and clinical management "
        "using standard medical terminology as found in hospital documentation."
    ),
}

SYSTEM_PROMPT = (
    "You are a clinical documentation specialist creating de-identified synthetic "
    "medical records for an ICD-10 coding research dataset. All records are "
    "fictional and used solely for training automated coding models.\n\n"
    "Generate a single realistic SOAP note for the given ICD-10 code. "
    "Use different demographics and clinical details than the examples provided. "
    "Do NOT include the ICD-10 code string anywhere in the note. "
    "Each section should be 2-4 sentences."
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(backend: str, lmstudio_url: str, lmstudio_model: str):
    """Create a Pydantic AI agent for the chosen backend."""
    from pydantic_ai import Agent

    if backend == "lmstudio":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        model = OpenAIChatModel(
            lmstudio_model,
            provider=OpenAIProvider(
                base_url=f"{lmstudio_url}/v1",
                api_key="lm-studio",
            ),
        )
    else:
        # Anthropic — reads ANTHROPIC_API_KEY from environment automatically
        from pydantic_ai.models.anthropic import AnthropicModel
        model = AnthropicModel("claude-sonnet-4-6")

    return Agent(
        model,
        output_type=SoapNote,
        system_prompt=SYSTEM_PROMPT,
        retries=3,
    )


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------

async def generate_note(
    agent,
    code: str,
    chapter: str,
    examples: list[dict],
) -> SoapNote | None:
    """Generate one SOAP note for the given code. Returns None on failure."""
    ex_text = ""
    for i, ex in enumerate(examples[:2], 1):
        ex_text += (
            f"\nExample {i}:\n"
            f"S: {(ex.get('subjective') or '')[:150]}\n"
            f"A: {(ex.get('assessment') or '')[:150]}\n"
        )

    prompt = (
        f"ICD-10 Code: {code}\n"
        f"Chapter: {CHAPTER_CONTEXT.get(chapter, '')}\n"
        f"{ex_text}\n"
        f"Generate one de-identified synthetic SOAP note for code {code}. "
        f"Do not mention the code itself."
    )

    try:
        result = await agent.run(prompt)
        return result.output
    except Exception as e:
        print(f"        Agent error ({type(e).__name__}): {str(e)[:200]}")
        return None


async def generate_n_notes(
    agent,
    code: str,
    chapter: str,
    examples: list[dict],
    n: int,
    concurrency: int = 3,
) -> list[SoapNote]:
    """Generate n notes concurrently with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def _one():
        async with sem:
            return await generate_note(agent, code, chapter, examples)

    tasks = [_one() for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Schema-safe row builder
# ---------------------------------------------------------------------------

def build_new_rows(
    notes: list[SoapNote],
    code: str,
    start_id: int,
    gold_schema: dict[str, Any],
) -> pl.DataFrame:
    """
    Build a DataFrame of new rows matching the gold layer schema exactly.
    Reads the schema to cast every column to the correct type.
    """
    rows = []
    for i, note in enumerate(notes):
        new_id = start_id + i
        full_note = (
            f"Subjective: {note.subjective}\n\n"
            f"Objective: {note.objective}\n\n"
            f"Assessment: {note.assessment}\n\n"
            f"Plan: {note.plan}"
        )
        # label column is List(String) in the gold layer
        label_val = [code] if isinstance(gold_schema.get("label"), pl.List) else code
        rows.append({
            "id":                  str(new_id),
            "note":                full_note,
            "dialogue":            "",
            "label":               label_val,
            "subjective":          note.subjective,
            "objective":           note.objective,
            "assessment":          note.assessment,
            "plan":                note.plan,
            "raw_code":            code.replace(".", ""),
            "standard_icd10":      code,
            "code_status":         "billable",
            "apso_note":           "",
            "apso_token_estimate": 0,
        })

    if not rows:
        return pl.DataFrame()

    new_df = pl.DataFrame(rows)

    # Cast each column to match gold schema exactly
    casts = []
    for col, dtype in gold_schema.items():
        if col in new_df.columns and new_df[col].dtype != dtype:
            casts.append(pl.col(col).cast(dtype))
    if casts:
        new_df = new_df.with_columns(casts)

    # Apply APSO pipeline
    new_df = build_apso_note(new_df)
    new_df = redact_icd10_sections(new_df)
    new_df = new_df.with_columns([
        (pl.col("apso_note").str.split(" ").list.len() * 1.3)
        .cast(gold_schema.get("apso_token_estimate", pl.Float64))
        .alias("apso_token_estimate")
    ])

    return new_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment weak ICD-10 chapters using Pydantic AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--chapters",        nargs="+", default=["Z", "O"])
    p.add_argument("--target",          type=int,  default=8)
    p.add_argument("--backend",         choices=["lmstudio", "anthropic"],
                   default="lmstudio")
    p.add_argument("--lmstudio-url",    default="http://localhost:1234")
    p.add_argument("--lmstudio-model",  default="qwen/qwen3.6-35b-a3b")
    p.add_argument("--concurrency",     type=int, default=3,
                   help="Notes generated in parallel per code (default: 3)")
    p.add_argument("--gold-path",       type=Path, default=None)
    p.add_argument("--output",          type=Path, default=None)
    p.add_argument("--dry-run",         action="store_true")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    # Resolve paths
    if args.gold_path:
        gold_path = args.gold_path
    else:
        gold_dir = config.resolve_path("data", "gold")
        parquets = sorted(
            p for p in gold_dir.glob("*.parquet") if "augmented" not in p.name
        )
        if not parquets:
            print(f"Error: no gold parquet in {gold_dir}", file=sys.stderr)
            sys.exit(1)
        gold_path = parquets[-1]

    output_path = args.output or gold_path.parent / "medsynth_gold_augmented.parquet"

    print(f"\n{'='*62}")
    print(f"  augment.py — Targeted Chapter Augmentation (Pydantic AI)")
    print(f"  Gold layer:  {gold_path.name}")
    print(f"  Output:      {output_path.name}")
    print(f"  Chapters:    {', '.join(args.chapters)}")
    print(f"  Target:      {args.target} records/code")
    print(f"  Backend:     {args.backend}")
    if args.backend == "lmstudio":
        print(f"  LM Studio:   {args.lmstudio_url}  ({args.lmstudio_model})")
    print(f"  Concurrency: {args.concurrency} notes/code")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"{'='*62}\n")

    # Load gold layer and capture schema
    gold_df    = pl.read_parquet(gold_path)
    gold_schema = dict(gold_df.schema)
    print(f"   Loaded gold layer: {len(gold_df):,} records")

    # Load or initialise working dataframe
    if output_path.exists() and output_path != gold_path:
        working_df = pl.read_parquet(output_path)
        already    = len(working_df) - len(gold_df)
        if already > 0:
            print(f"   ♻️  Resuming — {already:,} synthetic records already saved")
    else:
        working_df = gold_df.clone()

    if args.dry_run:
        for chapter in args.chapters:
            ch_df  = working_df.filter(
                pl.col("standard_icd10").str.starts_with(chapter) &
                (pl.col("code_status") == "billable")
            )
            counts = ch_df.group_by("standard_icd10").agg(pl.len().alias("n"))
            below  = counts.filter(pl.col("n") < args.target)
            to_gen = int((args.target - below["n"]).sum())
            print(f"   Chapter {chapter}: {len(below)} codes — would generate {to_gen} records")
        print("\n[DRY RUN complete — no files written]")
        return

    # Build agent
    agent = make_agent(args.backend, args.lmstudio_url, args.lmstudio_model)

    max_id       = int(working_df["id"].cast(pl.Int64).max())
    total_gen    = 0
    total_skip   = 0
    t0           = time.time()

    for chapter in args.chapters:
        print(f"\n── Chapter {chapter} {'─'*(50-len(chapter))}")

        ch_df = working_df.filter(
            pl.col("standard_icd10").str.starts_with(chapter) &
            (pl.col("code_status") == "billable")
        )
        counts = (
            ch_df.group_by("standard_icd10")
            .agg(pl.len().alias("n"))
            .sort("standard_icd10")
        )
        codes_needed = [
            (r["standard_icd10"], r["n"])
            for r in counts.filter(pl.col("n") < args.target).iter_rows(named=True)
        ]

        if not codes_needed:
            print(f"   ✅ All codes already at target")
            continue

        print(f"   📊 {len(codes_needed)} codes need augmentation")

        for code, current_n in codes_needed:
            needed   = args.target - current_n
            examples = (
                working_df.filter(pl.col("standard_icd10") == code)
                .sample(min(2, current_n), seed=42)
                .select(["subjective", "objective", "assessment", "plan"])
                .to_dicts()
            )

            notes = await generate_n_notes(
                agent, code, chapter, examples, needed, args.concurrency
            )

            if not notes:
                print(f"      ⚠️  {code}: 0/{needed} generated")
                total_skip += 1
                continue

            new_df = build_new_rows(notes, code, max_id + 1, gold_schema)
            if len(new_df) == 0:
                print(f"      ⚠️  {code}: schema build failed")
                total_skip += 1
                continue

            max_id     += len(new_df)
            working_df  = pl.concat([working_df, new_df])

            # Incremental save after every code
            working_df.write_parquet(output_path)

            added  = len(new_df)
            total_gen += added
            status = "✅" if added == needed else "⚠️ "
            print(f"      {status} {code}: +{added}/{needed} ({current_n} → {current_n+added})")

    elapsed = time.time() - t0
    final_new = len(working_df) - len(gold_df)

    print(f"\n{'='*62}")
    print(f"  ✅ Augmentation complete")
    print(f"  Original:   {len(gold_df):,} records")
    print(f"  Generated:  {final_new:,} new records")
    print(f"  Total:      {len(working_df):,} records")
    print(f"  Skipped:    {total_skip} codes")
    print(f"  Output:     {output_path}")
    print(f"  Elapsed:    {elapsed/60:.1f} min")
    print(f"{'='*62}")
    print(f"""
Next — retrain Z and O resolvers:

  uv run python scripts/train.py \\
      --experiment E-005b_Augmented_ZO \\
      --mode hierarchical --stage 2 --code-filter billable \\
      --stage2-init outputs/evaluations/E-002_FullICD10_ClinicalBERT/model/model \\
      --chapters Z O --gold-path {output_path}
""")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()