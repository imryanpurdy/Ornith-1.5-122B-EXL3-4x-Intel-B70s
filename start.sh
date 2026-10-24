#!/usr/bin/env bash
# ============================================================================
# start.sh — Ornith-1.5-122B EXL3 (4 bpw, EAQv1) on 4x Intel Arc Pro B70, TP4 + expert parallel
#
#   git clone <this repo> && cd <this repo>
#   cp .env.example .env      # MODEL_DIR, and RUNTIME_IMAGE if the pinned image is not present
#   ./scripts/build-vxk.sh    # once: the vllm_xpu_kernels overlay (~2-2.5 h CPU-only at -j2, needs network)
#   ./start.sh
#
# Commands: start | stop | status | logs   (default: start)
#   ./start.sh --print     print the docker run line and the gate plan, run nothing
#   ./start.sh stop        remove the container
#   ./start.sh status      container + API state
#   ./start.sh logs        docker logs -f
#
# Path: prereqs -> kernels (cached) -> plugin dist -> docker run -> first-request guard -> READY gates.
#
# The serve line is the measured operating point, not a starting point: TP4 + expert parallel, fp8_e4m3 KV,
# gpu-memory-utilization 0.85, max-model-len 262144, max-num-batched-tokens 4096, max-num-seqs 16, MTP k=2,
# FULL_DECODE_ONLY with capture sizes 1..96, qwen3_xml tool parser, qwen3_effort reasoning parser + plugin,
# image 2 / video 0.
#
# WHY THE FIRST-REQUEST GUARD EXISTS (step 4): a cold FULL_DECODE_ONLY engine whose FIRST request is a long
# prefill (>= 29K tokens) dies with EXL3XPU_OPEN_CAPTURE — a leaked SYCL command-graph recording left on the
# worker queue by the capture phase. Three short requests first drain that queue, after which the first real
# request can be any length. So: three "Say OK." requests, max_tokens 8, before anything else touches the API.
#
# A guard failure tears the container down (it is not going to serve). A READY-gate failure leaves it up for
# inspection and exits non-zero.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RUN_DIR="$SCRIPT_DIR/.run"

info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()  { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*" >&2; exit 1; }
red()  { printf '\033[1;31m%s\033[0m\n' "$*"; }
log()  { mkdir -p "$RUN_DIR"; printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$RUN_DIR/start.log"; }

# absolute host path without requiring the path to exist (so --print works before anything is built)
abs_path() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$SCRIPT_DIR" "${1#./}" ;; esac; }

is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_num()    { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

CMD=start
PRINT=false
for arg in "$@"; do
  case "$arg" in
    start|stop|restart|status|logs) CMD="$arg" ;;
    --print) PRINT=true ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) err "unknown argument: $arg (try --help)" ;;
  esac
done
[[ "$CMD" == restart ]] && CMD=start

# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------
ENV_FILE=.env
if [[ "$PRINT" == true && ! -f .env ]]; then
  warn "no .env: printing with the .env.example values"
  ENV_FILE=.env.example
fi
[[ -f "$ENV_FILE" ]] || err ".env not found: cp .env.example .env and fill in MODEL_DIR"
set -a; . "./$ENV_FILE"; set +a

for v in MODEL_DIR RUNTIME_IMAGE TOOLCHAIN_IMAGE BUILD_DIR PORT SERVED_MODEL_NAME GPUS \
         GPU_MEM_UTIL MAX_NUM_SEQS MTP_K MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS; do
  [[ -n "${!v:-}" ]] || err ".env: $v is empty"
done
CONTAINER_NAME="${CONTAINER_NAME:-ornith-exl3}"
VXK_DIR="${VXK_DIR:-$BUILD_DIR/vxk}"
READY_WAIT="${READY_WAIT_SECONDS:-1800}"

is_posint "$PORT" && [[ "$PORT" -le 65535 ]] || err "PORT must be 1-65535 (got '$PORT')"
is_num "$GPU_MEM_UTIL" || err "GPU_MEM_UTIL must be a number (got '$GPU_MEM_UTIL')"
awk -v g="$GPU_MEM_UTIL" 'BEGIN{exit !(g>0 && g<=1)}' || err "GPU_MEM_UTIL must be in (0,1] (got '$GPU_MEM_UTIL')"
is_posint "$MAX_NUM_SEQS" || err "MAX_NUM_SEQS must be a positive integer (got '$MAX_NUM_SEQS')"
[[ "$MAX_NUM_SEQS" -le 16 ]] || err "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds the validated 16 (above that is untested)"
is_posint "$MAX_MODEL_LEN" || err "MAX_MODEL_LEN must be a positive integer (got '$MAX_MODEL_LEN')"
[[ "$MAX_MODEL_LEN" -le 262144 ]] || err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the validated 262144"
is_posint "$MAX_NUM_BATCHED_TOKENS" || err "MAX_NUM_BATCHED_TOKENS must be a positive integer"
is_num "$MTP_K" && [[ "$MTP_K" =~ ^[0-9]+$ ]] || err "MTP_K must be an integer (got '$MTP_K')"
[[ "$MTP_K" -le 2 ]] || err "MTP_K=$MTP_K exceeds the validated 2"
IFS=',' read -r -a CARDS <<< "$GPUS"
[[ "${#CARDS[@]}" -eq 4 ]] || err "GPUS must list exactly 4 card indices for TP4 (got '$GPUS')"
ZE_AFFINITY_MASK="$GPUS"

# ---------------------------------------------------------------------------
# status / logs / stop
# ---------------------------------------------------------------------------
running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }

if [[ "$CMD" == status ]]; then
  if running "$CONTAINER_NAME"; then
    echo "container : $(docker inspect -f '{{.State.Status}} (up since {{.State.StartedAt}})' "$CONTAINER_NAME")"
  else
    echo "container : NOT RUNNING"
  fi
  api=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)
  if [[ -n "$api" && "$api" == *"$SERVED_MODEL_NAME"* ]]; then
    echo "api       : READY on :$PORT ($SERVED_MODEL_NAME)"
  else
    echo "api       : not answering on :$PORT"
  fi
  echo "log       : docker logs $CONTAINER_NAME"
  exit 0
fi
if [[ "$CMD" == logs ]]; then exec docker logs -f "$CONTAINER_NAME"; fi
if [[ "$CMD" == stop ]]; then
  info "=== stop ==="
  if running "$CONTAINER_NAME"; then docker rm -f "$CONTAINER_NAME" >/dev/null; ok "container $CONTAINER_NAME removed"; else info "not running"; fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Prerequisites — one clear failure line each, all before anything is touched
# ---------------------------------------------------------------------------
check_prereqs() {
  info "=== prerequisites ==="
  command -v docker >/dev/null 2>&1 || err "docker not found on PATH"
  docker info >/dev/null 2>&1 || err "docker daemon not reachable (is this user in the docker group?)"
  command -v curl >/dev/null 2>&1 || err "curl not found on PATH (used for the guard and READY gates)"
  PY="$(command -v python3 || true)"
  [[ -n "$PY" ]] || err "python3 not found on PATH (the READY gates are python3 scripts)"

  docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 \
    || err "RUNTIME_IMAGE is not present locally: $RUNTIME_IMAGE — see docs/runtime-image.md for what it is and how to obtain it (docker pull / docker buildx build)"
  ok "runtime image present: $RUNTIME_IMAGE"

  [[ -d "$MODEL_DIR" ]] || err "MODEL_DIR does not exist: $MODEL_DIR"
  [[ -f "$MODEL_DIR/config.json" ]] || err "MODEL_DIR has no config.json (wrong or torn weights tree): $MODEL_DIR"
  local shards
  shards=$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
  [[ "$shards" -ge 1 ]] || err "MODEL_DIR has no *.safetensors shards (wrong or torn weights tree): $MODEL_DIR"
  ok "weights: $MODEL_DIR ($shards shards)"

  # the port must be free before anything is launched: ANY listener fails, not just an engine-looking one
  # (timeout bounds the connect in case loopback traffic is ever filtered)
  if timeout 2 bash -c 'exec 3<>/dev/tcp/127.0.0.1/'"$PORT" 2>/dev/null; then
    err "port :$PORT already has a listener — free it first (ss -ltnp | grep ':$PORT')"
  fi
  # a card set is busy when another running container holds the render nodes
  local id nm devs
  for id in $(docker ps -q 2>/dev/null || true); do
    nm=$(docker inspect -f '{{.Name}}' "$id"); nm=${nm#/}
    devs=$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$id")
    case " $devs " in
      *"/dev/dri"*) [[ "$nm" == "$CONTAINER_NAME" ]] || err "cards busy: container '$nm' holds /dev/dri — stop it first (docker rm -f $nm)" ;;
    esac
  done
  ok "cards free: nothing on :$PORT, no other container holds /dev/dri"

  local b="$BUILD_DIR"
  if [[ ! -f "$b/lib/_C.so" || ! -f "$b/lib/_moe.so" ]]; then
    docker image inspect "$TOOLCHAIN_IMAGE" >/dev/null 2>&1 \
      || err "kernels not built ($b/lib is empty) and TOOLCHAIN_IMAGE is not present locally: $TOOLCHAIN_IMAGE — docker pull it, or point BUILD_DIR at an existing build"
  fi
  [[ -f "$VXK_DIR/vllm_xpu_kernels/_C.abi3.so" ]] \
    || err "vllm_xpu_kernels overlay not built: $VXK_DIR/vllm_xpu_kernels — run scripts/build-vxk.sh (docs/runtime-image.md §2.4); the engine will not start without it"
  ok "vllm_xpu_kernels overlay present: $VXK_DIR/vllm_xpu_kernels"
}

# ---------------------------------------------------------------------------
# Step 1 — EXL3 kernels (cached: skip when the outputs are newer than the sources)
# ---------------------------------------------------------------------------
step_kernels() {
  info "=== step 1: EXL3 kernels ==="
  local b="$BUILD_DIR" stale=0 f
  if [[ -f "$b/lib/_C.so" && -f "$b/lib/_moe.so" ]]; then
    for f in xpu/csrc/exl3_esimd.h xpu/csrc/exl3_ops.sycl xpu/csrc/exl3_moe.sycl xpu/csrc/fa_esimd.h xpu/csrc/ngram_nvme_xpu.sycl; do
      [[ "$f" -nt "$b/lib/_C.so" || "$f" -nt "$b/lib/_moe.so" ]] && stale=1
    done
    if [[ "$stale" -eq 0 ]]; then
      ok "cached: $b/lib/{_C.so,_moe.so} are newer than xpu/csrc"
      return 0
    fi
    info "xpu/csrc changed since the last build; rebuilding"
  fi
  ./scripts/build-kernels.sh all
  ok "kernels built: $b/lib/{_C.so,_moe.so}"
}

# ---------------------------------------------------------------------------
# Step 2 — plugin dist: what the container mounts at /opt/exl3xpu
#   exl3xpu/{*.py,_C.so,_moe.so} + exl3xpu-0.2.0.dev0.dist-info/{METADATA,entry_points.txt}
#   + templates/{ornith-chat-template.jinja,qwen3_effort_parser.py} + trellis_core/ (the split rules)
# The dist-info is what makes vLLM's general-plugins entry point find exl3xpu.vllm_plugin:register.
# ---------------------------------------------------------------------------
step_dist() {
  info "=== step 2: plugin dist (.run/dist) ==="
  local d="$RUN_DIR/dist" b
  b=$(abs_path "$BUILD_DIR")
  rm -rf "$d"
  mkdir -p "$d/exl3xpu" "$d/exl3xpu-0.2.0.dev0.dist-info" "$d/templates" "$d/trellis_core/format"
  cp xpu/exl3xpu/*.py "$d/exl3xpu/"
  cp "$b/lib/_C.so" "$b/lib/_moe.so" "$d/exl3xpu/"
  cp templates/ornith-chat-template.jinja templates/qwen3_effort_parser.py "$d/templates/"
  cp core/src/trellis_core/__init__.py core/src/trellis_core/reference.py "$d/trellis_core/"
  cp core/src/trellis_core/format/*.py "$d/trellis_core/format/"
  cat > "$d/exl3xpu-0.2.0.dev0.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: exl3xpu
Version: 0.2.0.dev0
Summary: EXL3 XPU kernels and vLLM plugin (Ornith)
EOF
  cat > "$d/exl3xpu-0.2.0.dev0.dist-info/entry_points.txt" <<'EOF'
[vllm.general_plugins]
exl3xpu = exl3xpu.vllm_plugin:register
EOF
  # content id over the sorted "<sha256>  <relpath>" listing, BUILD_INFO excluded (same rule as tools/make_dist.sh)
  local listing id
  listing=$(cd "$d" && LC_ALL=C find . -type f ! -name BUILD_INFO -printf '%P\n' | LC_ALL=C sort |
    while IFS= read -r r; do printf '%s  %s\n' "$(sha256sum "$r" | cut -d' ' -f1)" "$r"; done)
  id=$(printf '%s\n' "$listing" | sha256sum | cut -d' ' -f1 | cut -c1-12)
  {
    echo "id=$id"
    echo "commit=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "provenance: per-file sha256 of every file in this dist (BUILD_INFO itself excluded from the id)"
    printf '%s\n' "$listing"
  } > "$d/BUILD_INFO"
  DIST_DIR="$d"
  ok "dist id $id ($(printf '%s\n' "$listing" | wc -l) files)"
}

# ---------------------------------------------------------------------------
# Step 3 — the container
# ---------------------------------------------------------------------------
build_run() {
  local model vxk
  model=$(abs_path "$MODEL_DIR")
  vxk=$(abs_path "$VXK_DIR")

  SERVE_ARGS=(
    serve /models/ornith
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --quantization exl3
    --dtype bfloat16
    --tensor-parallel-size 4
    --enable-expert-parallel
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --limit-mm-per-prompt '{"image":2,"video":0}'
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,40,48,56,64,96]}'
    --cudagraph-metrics
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --generation-config auto
    --chat-template /opt/exl3xpu/templates/ornith-chat-template.jinja
    --reasoning-parser qwen3_effort
    --reasoning-parser-plugin /opt/exl3xpu/templates/qwen3_effort_parser.py
    --kv-cache-dtype fp8_e4m3
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}"
  )
  RUN_ARGS=(
    -d --name "$CONTAINER_NAME"
    --network host --ipc host --shm-size 17179869184
    --group-add render --security-opt label=disable
    --device /dev/dri:/dev/dri:rwm
    -v /dev/dri/by-path:/dev/dri/by-path
    -v "$RUN_DIR/cache:/cache"
    -v "$RUN_DIR/sycl-cache:/cache/sycl"
    -v "$model:/models/ornith:ro"
    -v "$DIST_DIR:/opt/exl3xpu:ro"
    -v "$vxk/vllm_xpu_kernels:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:ro"
    --env-file "$SCRIPT_DIR/env.list"
    -e "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
    -e EXL3_VLLM_PATCHES=fp8kv
  )
  # ornith.config: a hash of what decides what serves (env.list, the dist's contents, the image, the serve args).
  # It is this recipe's own config hash, not a deployment plan's.
  CONFIG_SHA=$(
    { cat "$SCRIPT_DIR/env.list"; cat "$DIST_DIR/BUILD_INFO" 2>/dev/null || true
      docker image inspect -f '{{.Id}}' "$RUNTIME_IMAGE" 2>/dev/null || printf '%s\n' "$RUNTIME_IMAGE"
      printf '%s\n' "${SERVE_ARGS[@]}"; } | sha256sum | cut -d' ' -f1)
  DOCKER_RUN=(docker run "${RUN_ARGS[@]}" --label "ornith.config=$CONFIG_SHA" --entrypoint /opt/venv/bin/vllm "$RUNTIME_IMAGE" "${SERVE_ARGS[@]}")
}

# print the array as one readable shell line: tokens with spaces get single quotes, everything else verbatim
print_cmd() {
  printf 'docker run'
  local a
  for a in "${DOCKER_RUN[@]:2}"; do
    case "$a" in
      ""|*[!A-Za-z0-9_./:=,@+-]*) printf " '%s'" "$a" ;;
      *) printf ' %s' "$a" ;;
    esac
  done
  echo
}

print_plan() {
  info "=== plan (--print: nothing is executed) ==="
  echo
  echo "# 1. prerequisites: .env complete, docker, python3, $RUNTIME_IMAGE present locally,"
  echo "#    $MODEL_DIR with config.json + shards, nothing on :$PORT, no other container on /dev/dri,"
  echo "#    $VXK_DIR/vllm_xpu_kernels present"
  echo "# 2. kernels: scripts/build-kernels.sh into $BUILD_DIR (skipped when the outputs are newer than xpu/csrc)"
  echo "# 3. dist: $RUN_DIR/dist from xpu/exl3xpu + $BUILD_DIR/lib/*.so + templates/ + core/src/trellis_core"
  echo "# 4. docker run (below), then wait for /v1/models and 'Application startup complete'"
  echo "# 5. first-request guard: 3 x 'Say OK.' (max_tokens 8) — a cold FULL_DECODE_ONLY engine whose first"
  echo "#    request is a >=29K-token prefill dies with EXL3XPU_OPEN_CAPTURE; three short requests drain the queue"
  echo "# 6. READY: nan sweep 14/14 (scripts/checks/nan_sweep.py) + tool calls (scripts/checks/toolcall.py)"
  echo
  print_cmd
}

# ---------------------------------------------------------------------------
# Step 4 — first-request guard
# ---------------------------------------------------------------------------
guard_fail() {
  red "FIRST-REQUEST GUARD FAILED: $*"
  warn "tearing the container down (it will not serve)"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | tail -40 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  log "GUARD_FAIL $*"
  exit 1
}

step_guard() {
  info "=== step 4: first-request guard (3 short requests) ==="
  local i resp code body n
  for i in 1 2 3; do
    resp=$(curl -sS -m 300 -w $'\n%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"temperature\":0}" 2>&1) \
      || guard_fail "request $i: curl failed ($(printf '%s' "$resp" | head -c 200))"
    code=${resp##*$'\n'}
    body=${resp%$'\n'*}
    [[ "$code" == 200 ]] || guard_fail "request $i: HTTP $code ($(printf '%s' "$body" | head -c 300))"
    n=$(printf '%s' "$body" | grep -o '"completion_tokens":[0-9]*' | head -1 | grep -o '[0-9]*$' || true)
    [[ -n "$n" && "$n" -ge 1 ]] || guard_fail "request $i: HTTP 200 but no completion tokens ($(printf '%s' "$body" | head -c 300))"
    ok "guard $i/3: HTTP 200, $n completion token(s)"
  done
  log "GUARD PASS"
}

# ---------------------------------------------------------------------------
# Step 5 — READY gates (failures leave the container up)
# ---------------------------------------------------------------------------
step_ready() {
  info "=== step 5: READY gates ==="
  local failed=0
  if "$PY" "$SCRIPT_DIR/scripts/checks/nan_sweep.py" --port "$PORT" --model "$SERVED_MODEL_NAME"; then
    ok "nan sweep PASS"
  else
    red "nan sweep FAIL (rows above)"
    failed=1
  fi
  if "$PY" "$SCRIPT_DIR/scripts/checks/toolcall.py" --port "$PORT" --model "$SERVED_MODEL_NAME" --count 5; then
    ok "tool calls PASS"
  else
    red "tool calls FAIL (rows above)"
    failed=1
  fi
  if [[ "$failed" -ne 0 ]]; then
    log "READY_FAIL"
    err "READY FAIL — container $CONTAINER_NAME is left running for inspection (docker logs $CONTAINER_NAME)"
  fi
  log "READY PASS"
  ok "READY — $SERVED_MODEL_NAME on :$PORT"
  info "  status: ./start.sh status   logs: ./start.sh logs   stop: ./start.sh stop"
}

# ---------------------------------------------------------------------------
# go
# ---------------------------------------------------------------------------
if [[ "$PRINT" == true ]]; then
  DIST_DIR="$RUN_DIR/dist"
  build_run
  print_plan
  exit 0
fi

check_prereqs
step_kernels
step_dist
build_run

info "=== step 3: launching $CONTAINER_NAME ($SERVED_MODEL_NAME on :$PORT) ==="
mkdir -p "$RUN_DIR/cache" "$RUN_DIR/sycl-cache"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
log "launch config=$CONFIG_SHA image=$RUNTIME_IMAGE model=$MODEL_DIR dist=$DIST_DIR"
print_cmd >> "$RUN_DIR/start.log"
"${DOCKER_RUN[@]}" >/dev/null
ok "container started: $CONTAINER_NAME (config $CONFIG_SHA)"

info "loading weights + compiling (~4-6 min) — polling /v1/models up to ${READY_WAIT}s"
DEADLINE=$(( $(date +%s) + READY_WAIT ))
while :; do
  running "$CONTAINER_NAME" || err "container exited during load: $(docker logs --tail 25 "$CONTAINER_NAME" 2>&1)"
  if curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"; then break; fi
  [[ "$(date +%s)" -gt "$DEADLINE" ]] && err "timed out after ${READY_WAIT}s waiting for /v1/models: $(docker logs --tail 25 "$CONTAINER_NAME" 2>&1)"
  sleep 15
done
ok "/v1/models answers"

info "waiting for the engine's startup receipt"
RECEIPT_DEADLINE=$(( $(date +%s) + 300 ))
while :; do
  if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Application startup complete"; then break; fi
  running "$CONTAINER_NAME" || err "container exited before the startup receipt: $(docker logs --tail 25 "$CONTAINER_NAME" 2>&1)"
  [[ "$(date +%s)" -gt "$RECEIPT_DEADLINE" ]] && err "READY gate: /v1/models answers but no 'Application startup complete' in the logs"
  sleep 10
done
ok "engine startup receipt confirmed"

step_guard
step_ready
