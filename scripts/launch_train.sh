#!/usr/bin/env bash
# Launch one FinScale training run on THIS node. For multi-node, run the same command on every node.
#
# Single node:
#   bash scripts/launch_train.sh configs/single-1n.yaml
# Two nodes (run on both, at the same time):
#   NNODES=2 NODE_RANK=0 HEAD_ADDR=<node0 IP> PORT=<allowed port> bash scripts/launch_train.sh configs/replicate-2n.yaml
#   NNODES=2 NODE_RANK=1 HEAD_ADDR=<node0 IP> PORT=<allowed port> bash scripts/launch_train.sh configs/replicate-2n.yaml
# Leave NODE_RANK unset to let torchrun's c10d rendezvous assign ranks.
#
# Optional env: NPROC_PER_NODE (default 1, GB10 has one GPU), NCCL_SOCKET_IFNAME, NCCL_IB_HCA,
#               RESULTS_DIR (default results/runs), NO_TELEMETRY=1
set -euo pipefail

CONFIG=${1:?usage: launch_train.sh <config.yaml>  (path relative to the repo root)}
cd "$(dirname "$0")/.."
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG (from $(pwd))" >&2; exit 2; }

NNODES=${NNODES:-1}
NPROC=${NPROC_PER_NODE:-1}
PORT=${PORT:-29400}
RUN=$(basename "$CONFIG" .yaml)
RUN_DIR=${RESULTS_DIR:-results/runs}/$RUN
HOST=$(hostname -s)${NODE_RANK:+-n$NODE_RANK}   # rank suffix keeps files apart when 2 "nodes" share a host (rehearsal)
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/nodes"

# LOCAL_MODEL_DIR=/tmp/model: copy the weights from shared storage to node-local disk first and train
# from there. On the fb-hack GB10s, loading straight from the NFS volume hung inside load_model for
# 25 min (memory reserved, GPU idle) although plain NFS reads run at ~1.5 GB/s; RunPod loaded from
# local disk in 12 s. The run name and results dir stay those of the original config.
# DO NOT use on the fb-hack GB10s: /tmp in a Fuzzball job is RAM (= GPU memory on GB10), so the 59 GB
# copy OOM-kills the trainer (SIGKILL in load_model). The fastload hook below fixes the slow load instead.
if [[ -n "${LOCAL_MODEL_DIR:-}" ]]; then
  SRC_MODEL=$(awk '/pretrained_model_name_or_path:/ {print $2; exit}' "$CONFIG")
  if [[ ! -f "$LOCAL_MODEL_DIR/.copy_done" ]]; then
    echo "[finscale] copying model $SRC_MODEL -> $LOCAL_MODEL_DIR"
    T0=$(date +%s); mkdir -p "$LOCAL_MODEL_DIR"
    cp -r "$SRC_MODEL"/. "$LOCAL_MODEL_DIR"/ && touch "$LOCAL_MODEL_DIR/.copy_done"
    echo "[finscale] model copy done in $(( $(date +%s) - T0 ))s ($(du -sh "$LOCAL_MODEL_DIR" | cut -f1))"
  fi
  # GB10 unified memory: the copy leaves ~59 GB of page cache that CUDA cannot use -> drop it now and
  # keep dropping it while the weights are read (see scripts/evict_page_cache.py).
  python3 scripts/evict_page_cache.py "$LOCAL_MODEL_DIR" "$SRC_MODEL"
  python3 scripts/evict_page_cache.py "$LOCAL_MODEL_DIR" --every 2 --for "${EVICT_SECONDS:-900}" \
    > "$RUN_DIR/logs/evict-$HOST.log" 2>&1 &
  LOCAL_CONFIG=$(mktemp /tmp/"$RUN".XXXX.yaml)
  sed "s#pretrained_model_name_or_path: .*#pretrained_model_name_or_path: $LOCAL_MODEL_DIR#" "$CONFIG" > "$LOCAL_CONFIG"
  cp "$CONFIG" "$RUN_DIR/config.yaml"   # keep the original next to the results
  CONFIG=$LOCAL_CONFIG
fi

# Hang diagnosis: every DUMP_STACKS_EVERY seconds, print all Python thread stacks to the log.
if [[ -n "${DUMP_STACKS_EVERY:-}" ]]; then
  DIAG=$(mktemp -d /tmp/diag.XXXX)
  printf 'import faulthandler, sys\nfaulthandler.dump_traceback_later(%s, repeat=True, file=sys.stderr)\n' \
    "$DUMP_STACKS_EVERY" > "$DIAG/sitecustomize.py"
  export PYTHONPATH="$DIAG${PYTHONPATH:+:$PYTHONPATH}"
fi

# Fast load (default on; FASTLOAD=0 to disable): the NeMo reader mmaps each shard and faults it in
# over NFS at ~25 MB/s (~49 min for the model). scripts/fastload/sitecustomize.py copies one shard at a
# time into RAM with a sequential read (~1 GB/s) and mmaps that instead: ~1 min, ~5 GB extra RAM.
# Must come after the DUMP_STACKS_EVERY block: it goes first on PYTHONPATH and chains to that one.
if [[ "${FASTLOAD:-1}" != "0" ]]; then
  export FASTLOAD_SRC=${FASTLOAD_SRC:-$(awk '/pretrained_model_name_or_path:/ {print $2; exit}' "$CONFIG")}
  export FASTLOAD_DIR=${FASTLOAD_DIR:-/tmp/fastload_shards}
  export PYTHONPATH="$(pwd)/scripts/fastload${PYTHONPATH:+:$PYTHONPATH}"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Everything is local: never call the Hub (venue internet is slow) and keep caches (and their file
# locks) on node-local disk, not on the NFS volume.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
       HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} HF_HOME=${HF_HOME:-/tmp/hf_home} \
       TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python3 scripts/node_manifest.py --out "$RUN_DIR/nodes/$HOST.json" --config "$CONFIG" || true

TPID=""
if [[ -z "${NO_TELEMETRY:-}" ]]; then
  python3 scripts/telemetry.py --output "$RUN_DIR/telemetry-$HOST.csv" --interval 2 >/dev/null 2>&1 &
  TPID=$!
fi
cleanup() { [[ -n "$TPID" ]] && kill "$TPID" 2>/dev/null || true; rm -rf "${FASTLOAD_DIR:-/tmp/fastload_shards}"; }
trap cleanup EXIT

if [[ "$NNODES" == "1" ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC")
else
  HEAD_ADDR=${HEAD_ADDR:?set HEAD_ADDR to node 0 IP for multi-node}
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
  if [[ -n "${NODE_RANK:-}" ]]; then
    LAUNCH=(torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC" --node-rank="$NODE_RANK"
            --master-addr="$HEAD_ADDR" --master-port="$PORT")
  else
    LAUNCH=(torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC" --rdzv-backend=c10d
            --rdzv-endpoint="$HEAD_ADDR:$PORT" --rdzv-id="$RUN")
  fi
fi

echo "[finscale] host=$HOST run=$RUN nnodes=$NNODES node_rank=${NODE_RANK:-auto} head=${HEAD_ADDR:-local} port=$PORT"
echo "[finscale] ${LAUNCH[*]} -m nemo_automodel.cli.app $CONFIG"
START=$(date +%s)
set +e
"${LAUNCH[@]}" -m nemo_automodel.cli.app "$CONFIG" 2>&1 | tee "$RUN_DIR/logs/$HOST.log"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
printf '{"host": "%s", "exit_code": %d, "wall_seconds": %d, "start_epoch": %d, "end_epoch": %d}\n' \
  "$HOST" "$RC" "$((END - START))" "$START" "$END" > "$RUN_DIR/nodes/$HOST.exit.json"
echo "[finscale] exit=$RC wall=$((END - START))s log=$RUN_DIR/logs/$HOST.log"

# The rank-0 node writes training.jsonl; summarize wherever it exists.
if [[ -f "$RUN_DIR/ckpt/training.jsonl" ]]; then
  python3 scripts/parse_train_log.py --run-dir "$RUN_DIR" || true
fi
exit "$RC"
