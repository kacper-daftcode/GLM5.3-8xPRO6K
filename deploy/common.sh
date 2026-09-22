#!/usr/bin/env bash
# Wspólne ustawienia dla testów modeli na 8x RTX PRO 6000 (pozostale GPU hosta, np. RTX 5090, sa pomijane)
# po UUID, bo indeksy przesuwaja sie po odpieciu/dodaniu karty.
# Ustawienia hosta (sciezki modeli, katalog cache) czyta z "$(dirname "$0")/env" (git-ignored; wzor: env.example).
[ -f "$(dirname "${BASH_SOURCE[0]}")/env" ] && source "$(dirname "${BASH_SOURCE[0]}")/env"
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}   # katalog na hoscie z cache torch.compile/Triton (podkatalog per konfiguracja)
GPUS="\"device=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep 'RTX PRO 6000' | cut -d, -f1 | paste -sd,)\""
PRO_IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep 'RTX PRO 6000' | cut -d, -f1 | paste -sd,)
PORT=${PORT:-8000}
COMMON_DOCKER_ARGS=(
  --runtime nvidia --gpus "$GPUS"
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1
  -e NCCL_P2P_DISABLE=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e TRITON_CACHE_DIR=/root/.cache/vllm/triton
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm"
  --restart no -d
)

# Zatrzymuje wszystko, co siedzi na 8 dużych GPU (prod GLM-5.2 i kontenery testowe)
stop_big_gpu_containers() {
  for c in glm52-nvfp4-prod glm53-nvfp4-prod glm53-nvfp4-test glm53-flash-test qwen38-flash-next-test; do
    if sudo docker ps -q -f name="^${c}$" | grep -q .; then
      echo "[$(date -u +%T)] stop $c"; sudo docker stop -t 60 "$c" >/dev/null
    fi
  done
  # kontenery testowe usuwamy (prod tylko zatrzymujemy, zeby dalo sie wrocic `docker start glm52-nvfp4-prod`)
  for c in glm53-nvfp4-test glm53-flash-test qwen38-flash-next-test; do
    sudo docker rm -f "$c" >/dev/null 2>&1 || true
  done
  # poczekaj az VRAM sie zwolni
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$PRO_IDX" | sort -n | tail -1)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 2
  done
  echo "[$(date -u +%T)] VRAM max used na dużych GPU: ${used} MiB"
}

wait_ready() { # $1 = timeout s, $2 = nazwa kontenera (wykrywa exit zamiast czekac w nieskonczonosc)
  local t=${1:-1800} c=${2:-} i st
  for i in $(seq 1 $((t/10))); do
    if curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" | grep -q 200; then
      echo "[$(date -u +%T)] READY po $((i*10)) s"; return 0
    fi
    if [ -n "$c" ]; then
      st=$(sudo docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo missing)
      if [ "$st" != "running" ]; then
        echo "[$(date -u +%T)] KONTENER $c NIE DZIALA (status=$st) po $((i*10)) s. Ostatnie logi:"
        sudo docker logs --tail 40 "$c" 2>&1 | grep -v '\^\^\^' | tail -25 | cut -c1-240
        return 1
      fi
    fi
    sleep 10
  done
  echo "[$(date -u +%T)] TIMEOUT czekania na /health"; return 1
}
