#!/bin/bash

# 停止所有应用的脚本

echo "🛑 停止所有 Qwen2.5 Agent 应用..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# 从保存的PID文件中读取并杀死进程
if [ -f "$LOG_DIR/pids.txt" ]; then
    while IFS= read -r pid; do
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            echo "🔄 停止进程 PID: $pid"
            kill "$pid" 2>/dev/null || true
            sleep 1

            # 如果进程仍在运行，强制杀死
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "   强制停止 PID: $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done < "$LOG_DIR/pids.txt"

    # 清除PID文件
    > "$LOG_DIR/pids.txt"
fi

# 备选方案：通过进程名称杀死
echo "🔄 通过进程名称清理..."

pkill -f "web_agent_with_skills.py" 2>/dev/null || true
pkill -f "session_viewer.py" 2>/dev/null || true
pkill -f "session_analyzer.py" 2>/dev/null || true

sleep 2

echo ""
echo "✅ 所有应用已停止"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 日志文件保留在: $LOG_DIR"
echo ""

