#!/bin/bash
# Usage:
#   bash start2.sh <env_name> <n>
#   bash start2.sh alfworld 8    # start 8 alfworld server (port 36001-36008)
#   bash start2.sh webshop 4     # start 4 webshop server (port 36001-36004)

ENV_NAME="$1"
NUM_INSTANCES="${2:-1}"
BASE_PORT=36001

if [ -z "$ENV_NAME" ]; then
    echo "Usage: $0 <env_name> <n>"
    echo "  env_name: webshop | alfworld | sciworld | searchqa"
    echo "  n: number of parallel instances (default: 1)"
    exit 1
fi

# ================= 环境映射 =================
declare -A ENV_MAP
ENV_MAP[webshop]="agentenv-webshop:webshop:"
ENV_MAP[alfworld]="agentenv-alfworld:alfworld:"
ENV_MAP[sciworld]="agentenv-sciworld:sciworld:"
ENV_MAP[searchqa]="agentenv-searchqa:searchqa:SEARCHQA_FAISS_GPU=True"

if [ -z "${ENV_MAP[$ENV_NAME]}" ]; then
    echo "[ERROR] Unknown env: $ENV_NAME"
    echo "  Available: webshop, alfworld, sciworld, searchqa"
    exit 1
fi

IFS=':' read -r conda_env cmd extra_env <<< "${ENV_MAP[$ENV_NAME]}"

# ================= 基础配置 =================
LOG_DIR="./env_logs/timestamp_$(date +%Y_%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "[ENV] env=$ENV_NAME, instances=$NUM_INSTANCES, ports=${BASE_PORT}-$((BASE_PORT + NUM_INSTANCES - 1))"
echo "[ENV] log dir: $LOG_DIR"

# CONDA_BASE=$(conda info --base)
CONDA_BASE=/root/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ================= 清除占用端口的进程 =================
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    port=$((BASE_PORT + i))
    pids=$(lsof -ti :"$port" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "[CLEANUP] Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
    fi
done

# ================= 启动服务 =================
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    port=$((BASE_PORT + i))
    log_file="$LOG_DIR/${ENV_NAME}_${port}.log"

    echo "[MANAGER] Launching $ENV_NAME on port $port..."
    (
        if [ -n "$extra_env" ]; then
            export "$extra_env"
        fi
        conda activate "$conda_env" && \
        nohup $cmd --host "0.0.0.0" --port "$port" > "$log_file" 2>&1 &
    ) &
done

# ================= 等待所有服务就绪 =================
wait_for_health() {
    local port=$1
    local name=$2
    local max_wait=1000
    local elapsed=0

    echo -n "[MANAGER] Waiting for $name:$port..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://0.0.0.0:${port}/" > /dev/null 2>&1; then
            echo " Ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo " FAILED (Timeout)"
    exit 1
}

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    port=$((BASE_PORT + i))
    wait_for_health "$port" "$ENV_NAME"
done

echo "[MANAGER] All $NUM_INSTANCES $ENV_NAME servers are ready."
exit 0
