#!/usr/bin/env python3
"""
serve.py — ICD-10 Model Server Entrypoint
==========================================
Starts the FastAPI prediction server. Models are loaded once at startup
and reused across all requests.

Usage
-----
    # Default: localhost:8000, threshold=0.7
    uv run python scripts/serve.py

    # Custom port and threshold
    uv run python scripts/serve.py --port 8080 --threshold 0.6

    # Auto-reload on code changes (development)
    uv run python scripts/serve.py --reload

    # Different experiment
    uv run python scripts/serve.py \\
        --experiment E-004a_Hierarchical_E002Init \\
        --stage1-experiment E-003_Hierarchical_ICD10

Once running, visit:
    http://localhost:8000/docs    — interactive Swagger UI
    http://localhost:8000/health  — liveness check
    http://localhost:8000/info    — server configuration

Example prediction request:
    curl -s -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"note": "S: Left knee pain. O: Swelling. A: OA. P: NSAIDs."}' \\
        | python3 -m json.tool
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the ICD-10 prediction model server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 to expose on LAN.",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload server on code changes (development only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1). "
             "Keep at 1 for MPS/CUDA — multiple workers each load all models.",
    )

    # Model options
    parser.add_argument(
        "--experiment",
        default="E-004a_Hierarchical_E002Init",
        help="Stage-2 experiment to serve (default: E-004a_Hierarchical_E002Init).",
    )
    parser.add_argument(
        "--stage1-experiment",
        default="E-003_Hierarchical_ICD10",
        help="Stage-1 experiment to serve (default: E-003_Hierarchical_ICD10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Default confidence threshold for auto-code decision (default: 0.7). "
             "Can be overridden per-request.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default number of top predictions to return (default: 5). "
             "Can be overridden per-request.",
    )

    args = parser.parse_args()

    # Build the app with the configured options
    from src.server import create_app
    app = create_app(
        experiment        = args.experiment,
        stage1_experiment = args.stage1_experiment,
        default_threshold = args.threshold,
        default_top_k     = args.top_k,
    )

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed.\n"
            "Install it with:  uv add uvicorn fastapi\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nStarting ICD-10 Model Server")
    print(f"  Host:       {args.host}:{args.port}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Docs:       http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        app,
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        workers = args.workers,
    )


if __name__ == "__main__":
    main()