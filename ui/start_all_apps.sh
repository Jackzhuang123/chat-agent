#!/bin/bash

# 启动所有应用的脚本
# 这个脚本会同时启动：
# 1. 主应用 (web_agent_with_skills.py) - 端口 7860
# 2. 会话查看器 (session_viewer.py) - 端口 7861
# 3. 高级分析 (session_analyzer.py) - 端口 7862

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 启动 Qwen2.5 Agent 完整系统..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 使用虚拟环境Python
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3.9"
if [ ! -f "$VENV_PYTHON" ]; then
    # 尝试找到其他版本的虚拟环境Python
    VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 未找到虚拟环境Python！"
    echo "请先创建虚拟环境和安装依赖："
    echo ""
    echo "  cd $PROJECT_DIR"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "📍 使用虚拟环境Python: $VENV_PYTHON"
echo ""

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "📁 日志目录: $LOG_DIR"
echo ""

# 启动函数
start_app() {
    local app_name=$1
    local app_file=$2
    local port=$3
    local log_file="$LOG_DIR/${app_name}.log"

    echo "🔄 启动 $app_name..."
    echo "   📝 日志: $log_file"
    echo "   🔌 端口: $port"

    nohup $VENV_PYTHON "$SCRIPT_DIR/$app_file" > "$log_file" 2>&1 &
    local pid=$!
    echo "   ✅ PID: $pid"
    echo ""

    # 保存PID以便后续杀死进程
    echo $pid >> "$LOG_DIR/pids.txt"
}

# 清除旧的PID文件
> "$LOG_DIR/pids.txt"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 启动顺序:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. 启动会话查看器 (依赖最少)
start_app "会话查看器" "session_viewer.py" "7861"
sleep 2

# 2. 启动高级分析 (依赖最少)
start_app "高级分析" "session_analyzer.py" "7862"
sleep 2

# 3. 启动主应用 (需要更长的启动时间用于模型加载)
echo "🔄 启动 主应用..."
echo "   📝 日志: $LOG_DIR/主应用.log"
echo "   🔌 端口: 7860"
echo "   ⏳ 正在加载模型，这可能需要30-60秒..."
echo ""

# 直接启动主应用并保持运行（在项目目录中运行以确保相对路径正确）
cd "$PROJECT_DIR"
nohup $VENV_PYTHON "$SCRIPT_DIR/web_agent_with_skills.py" > "$LOG_DIR/主应用.log" 2>&1 &
pid=$!
echo $pid >> "$LOG_DIR/pids.txt"
echo "   ✅ PID: $pid"
echo ""
cd - > /dev/null

# 等待更长时间让主应用完全启动和加载模型
sleep 10

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 所有应用已启动！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 访问地址:"
echo "   • 主应用: http://127.0.0.1:7860"
echo "   • 会话查看: http://127.0.0.1:7861"
echo "   • 高级分析: http://127.0.0.1:7862"
echo ""
echo "📁 日志文件位置: $LOG_DIR"
echo ""
echo "🛑 停止所有应用: bash stop_all_apps.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 保持脚本运行，显示应用状态
echo "📡 监控应用状态..."
echo ""

while true; do
    # 检查应用是否仍在运行
    all_running=true

    if ! pgrep -f "web_agent_with_skills.py" > /dev/null; then
        echo "⚠️  主应用已停止"
        all_running=false
    fi

    if ! pgrep -f "session_viewer.py" > /dev/null; then
        echo "⚠️  会话查看器已停止"
        all_running=false
    fi

    if ! pgrep -f "session_analyzer.py" > /dev/null; then
        echo "⚠️  高级分析已停止"
        all_running=false
    fi

    if [ "$all_running" = true ]; then
        echo -ne "✅ 所有应用运行中... $(date '+%H:%M:%S')\r"
    fi

    sleep 10
done

