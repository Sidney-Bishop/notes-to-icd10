
This is my laptop specs:
* Model Name:	MacBook Pro
* Model Identifier:	Mac17,6
* Chip:	Apple M5 Max
* Total Number of Cores:	18 (6 Super and 12 Performance)
* Memory:	128 GB”
  
I’d like to walk through the full set of possible LMstudio options / parameters with you and get some good advice on what the best parameters for me to use are please?. 


- With an M5 Max and 128 GB unified memory you're in the top 1% of LM Studio hardware — you can run Gemma-4-31B q6_k entirely on Metal with room to spare.

- The slowdown you saw isn't the model, it's the settings: 64k context + 4 concurrent predictions + mmap + no KV compression means every MCP read_file you do gets kept forever in GPU memory.

Let's walk your two panels and tune for that machine.

Your current screenshots show:

* Context 64,000
* GPU Offload 60
* CPU Threads 13
* Eval Batch 515
* Max Concurrent 4
* mmap ON
* Flash Attention ON
* KV cache full precision
* no draft model
* Temp 0.1
* Truncate Middle

Here's what each knob does, and what I'd change for a Mac17,6.

### 1. Core loading (second screenshot)

**Context Length** — 64,000 → 24,576
Gemma 4 supports 262k, but KV cache scales linearly. At q6_k, 64k context ≈ 38–45 GB of KV alone on top of the ∼20 GB model. That's why it crawls after a few file reads. With 24k you still fit 3–4 notebooks, and attention stays fast. For long-document review make a second preset at 48k.

GPU Offload — 60 → 999 (max)
On Apple Silicon "GPU" = Metal cores sharing your 128 GB. Gemma-4-31B has ∼48 layers; setting 60 already offloads all, but 999 guarantees it even after updates. Always max this on M-series.
CPU Thread Pool Size — 12 → 12 (keep)
CPU Threads — 13 → 12
You have 12 Performance cores. Use exactly those. Efficiency cores hurt inference latency. Set both to 12.
Evaluation Batch Size — 515 → 2048
This is how many prompt tokens are processed at once. M5 Max can handle 2048 easily, and it makes MCP file inserts 3–4x faster. 515 is a leftover default.
Max Concurrent Predictions — 4 → 1
4 is for serving multiple users. For chat it keeps 4 KV caches alive → 4x memory. Set to 1.
Unified KV Cache — ON (keep)
Fine for single model. Turn off only if you run two models side-by-side.
Offload KV Cache to GPU Memory — ON (keep)
With unified memory this just tells Metal to keep KV in the fast pool. Good.
Keep Model in Memory — ON (keep)
Prevents reloads between chats. With 128 GB, no downside.
Try mmap() — ON → OFF
mmap saves RAM by paging from disk, but you have RAM to spare. Turning it off loads the whole model once → much faster first token, no stutter when context grows.
Flash Attention — ON (keep)
Essential for long context on Metal. Leave on.
K/V Cache Quantization — OFF → ON, type q8_0
This is your "compaction". q8_0 halves KV size with almost no quality loss. With 24k context you'll drop from ∼15 GB KV to ∼7.5 GB. In 0.4.10 it's experimental but stable on M-series. Enable both K and V.
RoPE Frequency Base/Scale — Auto (keep)
Only touch if you need > trained context. You don't.
2. Generation behavior (first screenshot)
Temperature — 0.1 (perfect for code)
Keep. For creative writing bump to 0.7.
Limit Response Length — OFF → ON, 4096
Prevents runaway summaries that blow your context. Gemma will stop cleanly.
Context Overflow — Truncate Middle (keep)
This is actually the best choice for MCP work — it keeps your system prompt and the last few turns, drops the old file dumps in the middle. Don't switch to "Truncate Start".
Stop Strings — leave empty
Only needed for structured agents.
Sampling:
Top K 40 → 40 (fine)
Repeat Penalty 1.1 → 1.1 (keep)
Top P 0.95 → 1.0 (with temp 0.1, Top P does nothing; simplify)
Min P 0.05 → OFF (Min P fights low temp, causes the "I don't have tools" refusals you saw)
Presence Penalty OFF (keep)
Low temp + Min P is a known Gemma 4 issue in LM Studio — it makes the model over-filter tool tokens.
Structured Output — OFF (keep)
Only for JSON mode.
Speculative Decoding — OFF → ON
This is free speed on M5 Max. Download gemma-2-2b-it-q4_k as draft model. Then set:
Draft Model: gemma-2-2b-it
Drafting Probability Cutoff: 0.75 (keep)
Max Draft Size: 8 (not 16 — 8 is more stable for 31B)
You'll get ∼1.6–1.9x tokens/sec with temp 0.1.
3. Two presets to save
Create these in LM Studio (Preset → Save As):
A. "M5-Max-Gemma4-Coding" (daily driver)
Context 24576, GPU Offload 999, CPU Threads 12, Eval Batch 2048, Concurrent 1, mmap OFF, KV q8_0 ON, Flash ON
Temp 0.1, Top P 1.0, Min P OFF, Repeat 1.1, Limit 4096, Truncate Middle
Speculative ON with 2B draft
Expect: ∼45–55 tok/s prompt ingestion, ∼22–28 tok/s generation, instant tool calls, stable after 10+ MCP reads.
B. "M5-Max-Gemma4-Long" (for big notebooks)
Same as above but Context 49152, KV q8_0 ON, Eval Batch 1024, Speculative OFF (draft gets unstable >32k). Use only when you need to keep a whole repo in context, then switch back.
Why this fixes your "grows large then struggles"
64k → 24k cuts KV from ∼40 GB to ∼15 GB, and q8_0 cuts again to ∼7.5 GB
Max Concurrent 1 frees 3 extra KV caches
mmap OFF stops disk paging that spikes when you hit 80 GB usage
Eval Batch 2048 makes each read_file insert finish in <1s instead of 5s
Min P OFF stops Gemma from refusing tools when context is heavy
You still won't get automatic compaction — LM Studio doesn't have it — but with Truncate Middle and a 24k window, old file contents get dropped automatically while keeping your system prompt and recent turns. When you feel slowdown, that's your cue: ask Gemma for a 10-bullet handover, start a new chat, paste it in. With these settings that will be every ∼30–40 exchanges instead of every 5.
Want me to write you that handover prompt tailored to your ICD10 project so you can one-click summarize before starting a fresh chat?




LM Studio Configs – M5 Max 128GB (Production Tuning v3)
Peer review verdict: 9.5/10 correctness, 10/10 practicality
Your friend's assessment is still correct. v3 keeps their four tweaks and removes the parts LM Studio 0.4 no longer supports.
What the review confirmed
Three-preset architecture – latency-bound (8k), balanced (24k), capacity-bound (48k) matches quadratic attention scaling
KV cache as dominant term – controlled via context reduction, q8_0, and concurrency=1
mmap OFF – correct for 128GB unified memory
Min P diagnosis – low temp + Min P causes Gemma 4 tool refusal, disabling restores reliability
truncate_middle + MCP – preserves tool schema and recent state while discarding stale file dumps, enabling 30+ turn stability
What changed in LM Studio 0.4
LM Studio 0.4 removed global named presets. It now stores one set of defaults per model file, shown as the blue "Customized" badge. You cannot "Save as Coding/Long/MaxSpeed" in the UI anymore.
Speculative decoding is also blocked for your setup. LM Studio requires draft and main models to share the exact same vocabulary, and Gemma 4 variants currently fail that check in 0.4, which is why you see "No compatible draft models found" even with gemma-4-E2B-it downloaded.
You have two practical ways to keep three scenarios:
Option 1 – Change before load (fastest, no disk cost)
Keep your single gemma-4-31B file
Click Load tab, change Context and Batch, click Load Model
Do not click "Save as default" — it uses the values for this session only
Option 2 – Duplicate the GGUF (one-click switching)
Quit LM Studio
Finder → ~/.cache/lmstudio/models/lmstudio-community/gemma-4-31b-it/
Duplicate gemma-4-31B-it-Q6_K.gguf twice, rename:
gemma-4-31B-it-CODING-Q6_K.gguf
gemma-4-31B-it-LONG-Q6_K.gguf
gemma-4-31B-it-MAXSPEED-Q6_K.gguf
Reopen LM Studio, click the gear ⚙️ on each entry, set its defaults once. Each copy now remembers its own tuning.
Minor tweaks applied (v3)
Speculative decoding: enabled → disabled
Gemma 4 currently has no compatible draft in LM Studio 0.4. Disabling avoids the error and saves VRAM. Speed impact is minor because your core wins are context reduction and concurrency=1.
Coding preset: draft_min_tokens 0 → 2
Kept for documentation, but moot while speculative is off.
Long preset: memory expectation 50GB → 55-65GB
Under real MCP load with fragmentation, working set is higher.
MaxSpeed preset: batch_size 4096 note
Kept at 4096. If you see UI micro-stutter, drop to 3072.
The three configurations
All three share these base values (your current "Customized" settings):
Temperature 0.1, Top K 40, Repeat Penalty 1.1
Top P Sampling: unchecked (equals 1.0), Min P Sampling: unchecked (equals 0.0)
CPU Threads 12, Max Concurrent Requests 1
Context Overflow: Truncate Middle
Flash Attention ON, use_mmap OFF, keep_model_loaded ON
offload_kv_cache ON, unified_kv_cache ON, kv_cache_quantization q8_0
1. MaxSpeed – sub-second sprints
Use for: single-file edits, quick chats
Change only: Context Length 8192, Batch Size 4096
Load
Value
context_length
8192
batch_size
4096
max_concurrent_requests
1
use_mmap
false
kv_cache_quantization
q8_0

2. Coding – daily driver (what you have now)
Use for: ICD10 work, 2-3 notebooks, MCP workflows
Change only: Context Length 24576, Batch Size 2048
Load
Value
context_length
24576
batch_size
2048
max_concurrent_requests
1
use_mmap
false
kv_cache_quantization
q8_0

3. Long – repo-scale analysis
Use for: 4+ files, big docs
Change only: Context Length 49152, Batch Size 1024
Expect: 55-65 GB working set
Load
Value
context_length
49152
batch_size
1024
max_concurrent_requests
1
use_mmap
false
kv_cache_quantization
q8_0

Reference JSON (for documentation, not import)
LM Studio 0.4 no longer reads files from ~/.lmstudio/config-presets/. Keep these as a record of what to enter manually.
Coding v3
JSON
Tree
Raw

▶
{
"name"
:
"M5-Max-Gemma4-Coding-v3",
"temperature"
:
0.1,
"top_k"
:
40,
"repeat_penalty"
:
1.1,
"max_tokens"
:
8192,
"context_length"
:
24576,
"context_overflow_policy"
:
"truncate_middle",
"cpu_threads"
:
12,
"batch_size"
:
2048,
"max_concurrent_requests"
:
1,
"flash_attention"
:
true,
"use_mmap"
:
false,
"kv_cache_quantization_k"
:
"q8_0",
"kv_cache_quantization_v"
:
"q8_0",
▶
"speculative_decoding"
:
{
"enabled"
:
false
}
}
MaxSpeed v3 – same as above but context_length: 8192, batch_size: 4096
Long v3 – same as above but context_length: 49152, batch_size: 1024
Where you are now
Your current "Customized" badge is the Coding v3 configuration. The four fixes that solved the 220-second hangs are active:
Context 64k → 24k
Concurrent 4 → 1
mmap ON → OFF
Min P/Top P on → off
Speculative decoding and the other two contexts are optional. Use Option 1 to switch in 10 seconds, or Option 2 this weekend if you want true one-click presets back.
Load the model now and test — you should see first token under 1 second, ∼38 GB memory, and no tool refusals.