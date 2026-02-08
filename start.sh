#!/bin/bash
# 启动多环境服务器脚本
#   ./start_servers.sh                    # 启动所有
#   ./start_servers.sh webshop            # 只启动 webshop
#   ./start_servers.sh webshop,alfworld   # 启动 webshop 和 alfworld (逗号分隔)
# 可用环境: webshop, alfworld, sciworld, searchqa
# ================= 参数解析 =================
# 直接获取第一个参数。
# 如果运行 `bash start_servers.sh webshop`，这里 TARGETS 就是 "webshop"
# 如果运行 `bash start_servers.sh`，这里 TARGETS 就是 空字符串
TARGETS="$1"

# ================= 基础配置 =================
LOG_DIR="./env_logs/timestamp_$(date +%Y_%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "   Starting Multi-Environment Servers   "
if [ -n "$TARGETS" ]; then
    echo "   Targets: $TARGETS"
else
    echo "   Targets: ALL (Default)"
fi
echo "   Logs:    $LOG_DIR"
echo "========================================"

pids=()

# ================= 核心逻辑 =================
cleanup() {
    echo -e "\n\n[MANAGER] Received shutdown signal. Cleaning up..."
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    wait
    echo "[MANAGER] All servers stopped."
}

trap cleanup SIGINT SIGTERM EXIT

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

free_port() {
    local port=$1
    local port_pids=""

    if ! command -v lsof >/dev/null 2>&1; then
        echo "[MANAGER] Error: lsof is required to free ports but was not found."
        exit 1
    fi

    port_pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$port_pids" ]; then
        echo "[MANAGER] Port $port in use by PID(s): $port_pids. Killing..."
        kill $port_pids 2>/dev/null
    fi
}

launch_server() {
    local env_name=$1
    local cmd=$2
    local host=$3
    local port=$4
    local name=$5
    local extra_env=$6
    local log_file="$LOG_DIR/${name}_${port}.log"

    free_port "$port"
    echo "[MANAGER] Launching $name..."
    (
        if [ -n "$extra_env" ]; then
            export "$extra_env"
        fi
        conda activate "$env_name" && \
        exec $cmd --host "$host" --port "$port"
    ) > "$log_file" 2>&1 &

    pids+=($!)
    echo "[MANAGER] $name started with PID $!"
}

wait_for_health() {
    local host=$1
    local port=$2
    local name=$3
    local max_wait=180
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://${host}:${port}/health" > /dev/null 2>&1; then
            echo "[MANAGER] $name is healthy."
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "[MANAGER] WARNING: $name did not become healthy within ${max_wait}s."
    return 1
}

should_start() {
    local env=$1
    # 1. 如果 TARGETS 为空，默认启动所有
    if [ -z "$TARGETS" ]; then
        return 0
    fi
    # 2. 检查 TARGETS 中是否包含某个环境
    if [[ ",$TARGETS," == *",$env,"* ]]; then
        return 0
    fi
    return 1
}

# ================= 启动服务 =================
# 格式: name:conda_env:cmd:port[:extra_env]
SERVICES=(
    "webshop:agentenv-webshop:webshop:36001:"
    "alfworld:agentenv-alfworld:alfworld:36002:"
    "sciworld:agentenv-sciworld:sciworld:36003:"
    "searchqa:agentenv-searchqa:searchqa:36004:SEARCHQA_FAISS_GPU=True"
)

for svc in "${SERVICES[@]}"; do
    IFS=':' read -r name env cmd port extra <<< "$svc"
    if should_start "$name"; then
        launch_server "$env" "$cmd" "0.0.0.0" "$port" "$name" "$extra"
        wait_for_health "0.0.0.0" "$port" "$name"
    fi
done

# ================= 守护进程 =================
echo -e "\n[MANAGER] Services running. Press Ctrl+C to stop."
wait
