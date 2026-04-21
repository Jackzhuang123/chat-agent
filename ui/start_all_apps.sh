#!/bin/bash

# 启动主应用的脚本
# 仅启动 web_agent_with_skills.py - 端口 7860

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
STARTUP_LOG="$LOG_DIR/startup.log"

mkdir -p "$LOG_DIR"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$STARTUP_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$STARTUP_LOG" >&2
}

find_venv_python() {
    local candidate
    for candidate in "$PROJECT_DIR/.venv/bin/python3.9" "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/.venv/bin/python"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

port_is_listening() {
    local port=$1
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_port() {
    local port=$1
    local retries=$2
    local delay=$3
    local i
    for ((i=1; i<=retries; i++)); do
        if port_is_listening "$port"; then
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

stop_existing_apps() {
    if [ -f "$LOG_DIR/pids.txt" ]; then
        while IFS= read -r pid; do
            if [ -n "$pid" ]; then
                kill "$pid" >/dev/null 2>&1 || true
            fi
        done < "$LOG_DIR/pids.txt"
    fi
    pkill -f "web_agent_with_skills.py" >/dev/null 2>&1 || true
    for port in 7860; do
        lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | xargs kill >/dev/null 2>&1 || true
    done
    : > "$LOG_DIR/pids.txt"
    sleep 2
}

start_app() {
    local app_name=$1
    local app_file=$2
    local port=$3
    local wait_retries=$4
    local log_file="$LOG_DIR/${app_name}.log"

    echo "🔄 启动 $app_name..."
    echo "   📝 日志: $log_file"
    echo "   🔌 端口: $port"
    : > "$log_file"

    (
        cd "$PROJECT_DIR" &&
        nohup "$VENV_PYTHON" "$SCRIPT_DIR/$app_file" > "$log_file" 2>&1 &
        echo $! > "$LOG_DIR/.last_pid"
    )
    local pid
    pid="$(cat "$LOG_DIR/.last_pid")"
    rm -f "$LOG_DIR/.last_pid"
    echo "$pid" >> "$LOG_DIR/pids.txt"
    echo "   ✅ PID: $pid"

    if wait_for_port "$port" "$wait_retries" 1; then
        echo "   ✅ 端口 $port 已监听"
        log_info "${app_name} 启动成功，PID=${pid}，port=${port}"
    else
        echo "   ❌ 端口 $port 未监听"
        echo "   📌 请查看日志: $log_file"
        log_error "${app_name} 启动失败，PID=${pid}，port=${port}，日志=${log_file}"
        return 1
    fi

    echo ""
    return 0
}

echo "🚀 启动 Qwen2.5 Agent 主应用..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

log_info "开始启动 QwenAgent 系统"

if ! VENV_PYTHON="$(find_venv_python)"; then
    echo "❌ 未找到虚拟环境 Python"
    echo "请先在 $PROJECT_DIR 下创建 .venv 并安装依赖"
    exit 1
fi

echo "📍 使用虚拟环境Python: $VENV_PYTHON"
echo "📁 日志目录: $LOG_DIR"
echo ""

stop_existing_apps

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 启动检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

start_app "主应用" "web_agent_with_skills.py" "7860" 60 || exit 1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 启动校验完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 访问地址:"
echo "   • 主应用: http://127.0.0.1:7860"
echo ""
echo "📁 日志文件位置: $LOG_DIR"
echo "🛑 停止所有应用: bash stop_all_apps.sh"
echo ""
echo "说明:"
echo "   • 本脚本现在只负责启动主应用并验证 7860 端口。"
echo "   • 若后续聊天无响应，请优先查看 $PROJECT_DIR/logs/monitor.log 中是否出现 request_received。"
