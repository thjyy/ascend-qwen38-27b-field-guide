#!/usr/bin/env bash
set -euo pipefail

# Public example only. Override these values for your environment.
IMAGE="${QWEN_IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0}"
MODEL_PATH="${QWEN_MODEL_PATH:-/srv/models/Qwen3.8-27B-w8a8}"
CONTAINER_NAME="${QWEN_CONTAINER_NAME:-qwen38-vllm}"
VISIBLE_DEVICES="${QWEN_VISIBLE_DEVICES:-8,9}"
PORT="${QWEN_PORT:-7001}"
HOST_NIC="${HOST_NIC:-bond0}"

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container already exists: %s\n' "$CONTAINER_NAME" >&2
  printf 'Stop, rename, or remove it explicitly before retrying.\n' >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --privileged \
  --shm-size 128g \
  --health-cmd='curl -fsS "http://127.0.0.1:${QWEN_PORT}/health" || exit 1' \
  --health-start-period=600s \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=5 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -e ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  -e QWEN_PORT="$PORT" \
  -e HCCL_BUFFSIZE=512 \
  -e HCCL_CONNECT_TIMEOUT=1800 \
  -e HCCL_SOCKET_IFNAME="$HOST_NIC" \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e TASK_QUEUE_ENABLE=1 \
  -v "$MODEL_PATH:/models/qwen38:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  --entrypoint bash \
  "$IMAGE" \
  -lc '
exec vllm serve /models/qwen38 \
  --host 0.0.0.0 \
  --port "${QWEN_PORT}" \
  --served-model-name qwen3.8 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --quantization ascend \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3,\"enforce_eager\":true}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
  --additional-config "{\"enable_cpu_binding\":true}" \
  --generation-config auto \
  --override-generation-config "{\"max_new_tokens\":32768}"
'

printf 'Started %s on port %s with visible devices %s.\n' \
  "$CONTAINER_NAME" "$PORT" "$VISIBLE_DEVICES"
printf 'Follow startup: docker logs -f %s\n' "$CONTAINER_NAME"

