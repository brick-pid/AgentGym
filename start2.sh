#!/bin/bash
# start2.sh
# start2.sh alfworld
# start2.sh alfworld,sciworld

TARGETS="$1"

# ================= 基础配置 =================
LOG_DIR="./env_logs/timestamp_$(date +%Y_%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "[ENV] 开始初始化环境..."
echo "[ENV] 日志目录: $LOG_DIR"

# CONDA_BASE=$(conda info --base)
CONDA_BASE=/root/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ================= 核心函数 =================

launch_server() {
    local env_name=$1
    local cmd=$2
    local host=$3
    local port=$4
    local name=$5
    local extra_env=$6
    local log_file="$LOG_DIR/${name}_${port}.log"

    echo "[MANAGER] Launching $name on port $port..."
    
    # 使用 nohup ... & 放入后台，且脚本退出后不杀进程
    (
        if [ -n "$extra_env" ]; then
            export "$extra_env"
        fi
        conda activate "$env_name" && \
        nohup $cmd --host "$host" --port "$port" > "$log_file" 2>&1 &
    ) &
}

wait_for_health() {
    local host=$1
    local port=$2
    local name=$3
    local max_wait=300 # 等待时间，单位秒
    local elapsed=0

    echo -n "[MANAGER] Waiting for $name..."
    while [ $elapsed -lt $max_wait ]; do
        # 只要 health 接口通了，就返回
        if curl -sf "http://${host}:${port}/health" > /dev/null 2>&1; then
            echo " Ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo " FAILED (Timeout)"
    exit 1 # 如果起不来，直接报错退出，阻断后续训练
}

should_start() {
    local env=$1
    if [ -z "$TARGETS" ]; then return 0; fi
    if [[ ",$TARGETS," == *",$env,"* ]]; then return 0; fi
    return 1
}

# ================= 启动流程 =================
SERVICES=(
    "webshop:agentenv-webshop:webshop:36001:"
    "alfworld:agentenv-alfworld:alfworld:36002:"
    "sciworld:agentenv-sciworld:sciworld:36003:"
    "searchqa:agentenv-searchqa:searchqa:36004:SEARCHQA_FAISS_GPU=True"
)

# 1. 启动所有目标服务
for svc in "${SERVICES[@]}"; do
    IFS=':' read -r name env cmd port extra <<< "$svc"
    if should_start "$name"; then
        launch_server "$env" "$cmd" "0.0.0.0" "$port" "$name" "$extra"
    fi
done

# 2. 只有当服务都 Ready 了，脚本才结束
for svc in "${SERVICES[@]}"; do
    IFS=':' read -r name env cmd port extra <<< "$svc"
    if should_start "$name"; then
        wait_for_health "0.0.0.0" "$port" "$name"
    fi
done

echo "[MANAGER] All requested servers are ready. Exiting setup script."
# 脚本结束，Server 继续在后台运行
exit 0