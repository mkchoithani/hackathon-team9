# Project context (carried over from the pre-event Claude session)

## Project brief

You are helping me build a hackathon project FROM SCRATCH in a new, empty repo. All code must be
written now, during the event. Do not ask me for old code. Work in small verified steps, and after
each step tell me the exact command to run and what a good result looks like.

# 1. Event and goal
- CIQ Supercomputing Local AI Hackathon (Chicago, Nov 13–15). Track: MULTI-NODE TRAINING.
- Hardware: 2× Lenovo ThinkStation PGX, each 1× NVIDIA GB10 (ARM64 Grace CPU, Blackwell GPU, ~128 GB
  memory SHARED by CPU and GPU). Orchestrated by CIQ Fuzzball (v4 workflows). Judges want:
  (a) a real 2-node workload, (b) quantitative 1-node vs 2-node benchmarks, (c) runs executed through
  Fuzzball, (d) proof of what the 2nd node unlocks (throughput, time-to-convergence, or memory/capacity).
- Product "FinScale": fine-tune NVIDIA Nemotron 3 Nano 30B-A3B (LoRA) to read one 10-K passage and
  return ONE JSON object: {"risk_class","severity","evidence_quote","rationale"}
  risk_class ∈ market, credit, liquidity, operational, regulatory_legal, cybersecurity, supply_chain, other
  severity ∈ low, medium, high. evidence_quote must be copied VERBATIM from the passage.

# 2. Pipeline to build (one small script per step, all paths relative to repo root)
1. Download latest 10-K per company from SEC EDGAR (30 large US tickers across sectors).
2. Extract sections Items 1A, 1C, 3, 7, 7A and cut into ~4,000-char passages (max ~40 per company,
   round-robin across sections, dedupe).
3. Label passages with a large "teacher" model via the NVIDIA API (OpenAI-compatible), validate.
4. Build train/val/test SPLIT BY COMPANY, as chat rows {"messages":[system,user,assistant]}.
5. Token gate: prove every answer survives tokenization at the training length.
6. Generate NeMo AutoModel LoRA configs for: single (1 node), ddp (2 nodes), ep (2 nodes).
7. Launch training (1 node, then 2 nodes via Fuzzball), log per-step metrics.
8. Evaluate base vs fine-tuned on held-out companies; score; make charts.

# 3. Facts already verified — follow them, do not rediscover
## Data / SEC
- SEC requires header User-Agent "<project> <real email>"; ≤10 req/s (use ~0.2 s pause, retry 403/429/5xx).
- company_tickers.json → CIK; submissions JSON "filings.recent" only holds ~1,000 filings: ALSO scan
  older pages listed in "filings.files" for form == "10-K". Skip "10-K/A". Use primaryDocument URL.
- HTML→text: skip <script>, <style>, <head>, <title> and the hidden iXBRL <ix:header>; newline at block tags.
- Section finding: headings are lines starting "Item 1A" etc. Some filers repeat "Item 1A" as a page
  header on every page → MERGE consecutive headings of the same item; pick the LONGEST span per item
  (table of contents spans are tiny). Some filers (MS, INTC) have NO "Item" headings: fall back to
  matching whole-line section TITLES ("Risk Factors", "Cybersecurity", "Legal Proceedings",
  "Management's Discussion and Analysis…", "Quantitative and Qualitative Disclosures About (Market) Risk",
  plus non-target titles as boundaries). Handle ’ and ' apostrophes. Drop page numbers and short
  page-header lines. Known unfixable: WFC (risk text lives in Exhibit 13) — just skip it.
## Labeling (NVIDIA API, base URL https://integrate.api.nvidia.com/v1, key NVIDIA_API_KEY)
- FIRST write a --probe that sends a tiny request to candidate models and prints OK/FAIL per model.
  With my key: nvidia/nemotron-3-ultra-550b-a55b = OK (use as teacher); nvidia/nemotron-3-super-120b-a12b
  and nvidia/nemotron-nano-3-30b-a3b = 404. Being listed in /v1/models does NOT mean callable.
- Pass extra_body {"chat_template_kwargs":{"enable_thinking":false}}; strip <think>…</think> and code
  fences; extract the first JSON object.
- Validate every label: allowed class/severity (normalize synonyms like "regulatory"→regulatory_legal),
  quote ≥20 chars, ≤~400 chars, and present in the passage after whitespace/quote-mark normalization.
  On invalid, re-ask once with the reason ("quote too long" was the #1 failure).
- Resumable output file; 8 worker threads; STOP early if the first 5 calls all fail with 400/401/403/404;
  ALWAYS retry rows that failed with API errors on the next run; separate --retry-invalid pass.
- Expected: ~1,000 passages → ~95% valid labels; classes skew to regulatory_legal/market.
## Dataset
- Split by ticker (~15% test, ~10% val), cap "other" at ~25% of TRAIN only.
- The student system prompt (taxonomy + JSON-only instruction) must be IDENTICAL in training and eval.
- Write a manifest with counts, class balance and SHA-256 of each file; a gold_review.csv sample of
  the test set for humans to correct.
- Rows reach ~2,950 tokens (median ~1,100). Use seq_length 3072, not 2048.
## Tokenizer / chat template
- Nemotron 3 template renders a training row identically for enable_thinking True/False/unset:
  "<|im_start|>assistant\n<think></think>{answer}<|im_end|>". Inference with enable_thinking=False ends
  the prompt in "<think></think>", matching training. So DO NOT put chat_template_kwargs in the
  training config (AutoModel 0.7.0 rejects that field).
- Token gate must use AutoModel's own ChatDatasetConfig(path_or_dataset_id, seq_length, truncation=
  "longest_first").build(tokenizer) and require labels != -100 count to equal the untruncated count for
  EVERY row. (Historic bug: truncation removed the whole answer → zero supervised tokens.)
## Environment / training (NeMo AutoModel)
- Container: nvcr.io/nvidia/nemo-automodel:26.10.00 (has arm64 + amd64). Inside: AutoModel 0.7.0,
  transformers 5.18, torch 2.14 (CUDA 13.4), mamba_ssm 2.3.0, causal_conv1d 1.6.0, peft 0.20.
  Needs a CUDA ≥13 host driver (CUDA 13.2 hosts worked). Print these versions in a preflight script.
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (58.8 GiB, 13 safetensors shards). Download with
  huggingface_hub.snapshot_download; set HF_TOKEN. The download can look stalled (files not growing,
  0 MB/s) while it reassembles — it is NOT stuck; it took 24 min unauthenticated.
- Build configs by LOADING the installed recipe
  /opt/Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml and overriding:
  model path; dataset/validation_dataset = ChatDatasetConfig (fields above); validation_dataloader;
  step_scheduler {global_batch_size 16, local_batch_size 1, max_steps, val_every_steps, ckpt_every_steps};
  optimizer lr 1e-4; peft dim 16 alpha 32 (keep exclude_modules ["*.out_proj"]); dist_env timeout 30 min;
  checkpoint {enabled, checkpoint_dir, model_save_format: safetensors, save_consolidated: false}
  (adapter only). Remove the "ci" key. checkpoint_dir is ALSO where training.jsonl / validation.jsonl go.
- Distributed section per mode (GB10 = 1 GPU per node):
  single: strategy fsdp2, dp_size 1.
  ddp (main 2-node mode): strategy ddp, dp_size 2, tp/cp/ep 1, activation_checkpointing true.
    Do NOT use fsdp2 with dp_replicate_size == dp_size: AutoModel fails with "DDP usecase is not
    supported by FSDP2".
  ep (capacity mode): strategy fsdp2, dp_size 2, ep_size 2. shard: fsdp2 dp_size 2 (works but very
    network-bound; looks hung with GPUs at 0% — prefer ep).
- Keep GLOBAL batch identical across 1 vs 2 nodes (only per-rank grad accumulation changes) = fair compare.
- Launch: torchrun --nnodes=N --nproc-per-node=1 --node-rank=R --master-addr=HEAD --master-port=PORT
  -m nemo_automodel.cli.app <config.yaml> (single node: --standalone). Log each node to its own file
  named by hostname (+ rank suffix if two ranks share a host), record per-node manifest (hostname, IPs,
  GPU, versions, git sha, config sha, data hashes) and exit code, sample nvidia-smi + /proc/meminfo
  (GB10 unified memory: nvidia-smi memory may be N/A; host MemAvailable is the real signal).
- AutoModel log line: "step N | epoch | loss | grad_norm | lr | mem X GiB | tps A(B/gpu) | num_label_tokens
  | num_tokens_per_step | mfu"; training.jsonl has the same fields + timestamp. Parse it into steps.csv
  and summary.json; exclude the first 3 steps from throughput medians (step 0 ≈ 60 s warm-up).
- NCCL "WARN gin_plugin / libnccl-*.so not found" lines are harmless. For real errors, read ~30 lines
  ABOVE "torchrun … ChildFailedError".
## Reference results (RunPod, NOT GB10 — use to sanity-check only)
- 1× H100/H200: peak 62–64 GiB at seq 3072, ~2,200–2,550 tokens/s, loss 1.17 → 0.56 in 8 steps,
  val loss ~0.50. 2 ranks ddp: 4,855 tok/s = 1.90× (95% efficiency), same loss. ep: ~38 GiB/rank.
- Eval (60 held-out rows, 8 training steps): class acc 0.68→0.82, macro-F1 0.49→0.83,
  severity 0.47→0.70, JSON valid 0.98→1.00.
## Evaluation
- Load with transformers' BUILT-IN class (AutoModelForCausalLM, trust_remote_code=False →
  NemotronHForCausalLM). The repo's remote modeling_nemotron_h.py CRASHES in generate() on
  transformers 5.18 ("cache_position … NoneType"). Fall back to remote code only if no built-in class.
- Use dtype=torch.bfloat16 (fallback torch_dtype), device_map="cuda", apply_chat_template(…,
  add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt"), greedy.
- Adapter: PeftModel.from_pretrained(model, <checkpoint_dir>/LATEST/model). Base = same weights, no adapter.
- ~4 s/answer after warm-up. Metrics: JSON-valid rate, fully-valid rate, class accuracy, macro-F1 over
  classes present, severity accuracy, quote fidelity, median latency, confusion counts. Score ONLY rows
  every system answered. Optional --gold CSV with human corrections.
## Fuzzball (v4)
- Workflow: version: v4; volumes (persistent: use <provisioner>, name <volume>); job fields image.uri,
  script, env (list of KEY=VALUE), cwd (absolute), mounts {/path: volumeName}, resource {cpu cores,
  memory size, devices nvidia.com/gpu: 1, exclusive}, policy.timeout.execute.
- Multinode: multinode {nodes: 2, implementation: generic, procs-per-node: 1}; script sources
  /multinode/generic-helper, uses $MULTINODE_NODE_IP as head, exec_rank0 "<cmd rank 0>",
  exec_rankN "<cmd rank 1>", wait_rank0. Set MULTINODE_WRAPPER_FORWARD_STREAMS=1. Keep per-node commands
  simple: call a small entry script with (node_rank, head_ip, task).
- Job memory request must cover the unified-memory footprint (~64 GB model on GPU + CPU side): ask staff
  for the max; request ~100 GB.
- Put cluster-specific values (image, volume, port, NCCL_SOCKET_IFNAME) in one env file, never hardcode
  port 29500 (it timed out in a past attempt). CLI: fuzzball context login; fuzzball workflow start
  <file>.fz; fuzzball workflow log <id> <job> -f. MCP: fuzzball mcp serve (CLI ≥4.2).

# 4. Order of work (gate each step; stop and fix before moving on)
1. Repo skeleton, shared schema/prompt module, preflight script.
2. Data steps 1–5 on a laptop (CPU) → commit data/sft + manifest. (Ask organizers if pre-event data is
   allowed; otherwise rebuild here — ~45 min.)
3. Fuzzball: single-node smoke (preflight + 1 training step) on one GB10.
4. 2-node NCCL all-reduce test (correct sum, two distinct hostnames, bandwidth table).
5. 1-node baseline training run (≥3 repeats for benchmarks), save adapter.
6. 2-node ddp run, same GBS/seq/data (≥3 repeats) → speedup, efficiency, time-to-same-val-loss.
7. Capacity attempt: find a setting that OOMs on 1 node but works on 2 (e.g. ep with longer seq).
8. Eval base vs fine-tuned, charts (throughput, loss vs wall-clock, memory per node, NCCL bandwidth,
   eval quality), demo summary. Only claim what logs prove.

Start with step 1. Keep each script small, documented, and resumable. Ask me for cluster values
(Fuzzball endpoint, image, volume, port, NIC) when you need them.

# Event cluster facts (verified live)
- Fuzzball v4.3.1, cluster fb-hack, CLI at ~/bin/fuzzball, logged in as team9@fb-hack.local,
  ACTIVE GROUP = team9 (30741bdf-…). Claude has the Fuzzball MCP (read + write + exec).
- Team9 nodes: CIQ-CHI-SPARK17 (192.168.99.26) and SPARK18 (192.168.99.27): 1 GB10 GPU,
  20 ARM cores, 127.9 GiB unified memory each. Nodes can drop offline; check `fuzzball node list`.
- Storage: persistent volume `models` on provisioner `team9` (NFS 10.10.11.9:/hackathon/team-09,
  shared by both nodes). Cluster volume `datasets` also exists (read-only, CIQ's).
- HF token secret: secret://user/hf-token (type hf), verified working via hf:// ingress.
- Download the model with Fuzzball volume INGRESS (hf://nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16,
  secret secret://user/hf-token) onto `models`, once.
- Docs warn docker:// images >10 GB may fail OCI->SIF conversion. nemo-automodel:26.10.00 is not yet
  tested here. Run the image probe first; ask CIQ for a SIF/cached copy if the pull fails.
- CIQ's own 2-node example (catalog "Nemotron 3 Nano"): multinode generic, network host: true,
  exec_rank0 / exec_rankN / wait_rank0, $RANK expands per node, inter-node ports 13345 and 29501,
  NCCL settings via env (example NCCL_SOCKET_IFNAME=eth0; confirm the real NIC from the probe).
- Workflow outputs must be egressed to the Fuzzball object cache to be readable afterwards.

# Event plan

## Step 2: Setup run on the cluster (about 30–40 min, mostly waiting)
Ask the new session to run one setup workflow that:
1. downloads the 59 GB model onto `models` (ingress with `hf-token`);
2. runs the **image probe** on one GB10: chip, versions, memory, **network interface names**;
3. checks the model has all 13 files.

**Gate:** the versions match RunPod (AutoModel 0.7.0, transformers 5.18, mamba_ssm 2.3.0, …) and the GPU shows as `NVIDIA GB10`. If the image pull fails, ask CIQ for a SIF copy before going further.

## Step 3: Data (in parallel with step 2, 10 min)
Organizers allow your prepared work, so reuse the labeled dataset: copy `data/sft/` (and `data/labels.jsonl`, `data/passages.jsonl`) from the old repo into the new one, and commit it, noting in the message that it came from the pre-event rehearsal. If you'd rather keep everything fresh, rebuild with `REBUILD_DATA.md` (about 45 min).

## Step 4: Single-node training on GB10 (about 1 hour)
- **smoke:** 1 training step. Gate: finite loss, exit 0, memory about 64 GB.
- **1-node baseline:** 3 repeats, so you have proper benchmark numbers.

## Step 5: Two nodes (about 1–2 hours)
- **NCCL test across SPARK17 and SPARK18.** Gate: correct sum, two different hostnames, a bandwidth table.
- **2-node `ddp` training:** same batch, length and data as step 4, 3 repeats. This gives your **speedup** number on real GB10s.

## Step 6: Optional capacity attempt
`ep` mode with longer passages: find something that fails on 1 node and works on 2.

## Step 7: Results and demo (Sunday morning; deadline 4 PM CT)
- Evaluate before vs after training on the held-out companies.
- Charts: 1-node vs 2-node speed, loss over time, memory per node, NCCL bandwidth, quality.
- Slides covering three things: what you built, what the second node unlocked (with numbers), and that everything ran through Fuzzball. Submit early.

**Throughout:** keep logs, workflow IDs and screenshots of both nodes running. That's your proof for the judges.
