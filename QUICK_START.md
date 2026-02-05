# 🚀 快速开始指南

## 一键启动

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
./start_all.sh
```

系统会自动启动两个应用：
- **Chat Agent**: http://localhost:7860 (对话界面)
- **Session Viewer**: http://localhost:7861 (日志查看)

## 📱 应用功能

### Chat Agent (7860)
1. 在侧边栏配置模式和参数
2. 在对话框输入问题
3. 点击"发送"或按 Enter
4. 对话自动保存到日志

### Session Viewer (7861)

#### 📁 会话列表
- 查看所有保存的对话会话
- 点击选择会话查看详情
- 删除不需要的会话

#### 🔍 会话详情
1. 从会话列表选择一个会话
2. 点击"查看选中会话详情"
3. 查看统计信息（对话数、Token数等）
4. 查看完整的对话记录
5. 查看技能调用追踪

#### 💾 数据导出
- **导出会话**: 保存为 JSON 文件
- **清空日志**: 删除所有会话记录

## 📊 查看日志数据

### 方式 1: 通过 UI (推荐)
1. 打开 Session Viewer: http://localhost:7861
2. 在"会话列表"选择要查看的会话
3. 在"会话详情"中查看所有信息

### 方式 2: 直接查看文件
```bash
# 查看日志文件列表
ls -lah /Users/zhuangranxin/PyCharmProjects/chat-Agent/session_logs/

# 查看某个会话的内容
cat /Users/zhuangranxin/PyCharmProjects/chat-Agent/session_logs/20240204_152030_123.json | jq .
```

### 方式 3: 代码查询
```python
from ui.session_logger import get_logger

logger = get_logger()

# 列出所有会话
sessions = logger.get_all_sessions()
for s in sessions:
    print(f"{s['session_id']}: {s['message_count']} 条消息")

# 查看某个会话
data = logger.get_session_details("20240204_152030_123")
print(f"总对话数: {len(data['messages'])}")
print(f"总 Token: {data['statistics']['total_tokens_used']}")
```

## 🎯 常见操作

### 导出单个会话
1. 打开 Session Viewer
2. 进入"会话列表"标签
3. 选择要导出的会话
4. 切换到"数据导出"标签
5. 点击"导出会话"

### 删除旧日志
1. 打开 Session Viewer
2. 进入"数据导出"标签
3. 点击"清空所有日志"确认

### 分析对话数据
1. 打开 Session Viewer
2. 进入"会话详情"标签
3. 查看统计卡片了解全局信息
4. 浏览对话记录查看具体内容

## 💡 使用技巧

### 对话优化
- 使用"Skills 系统"获得更好的回答
- 上传 PDF 文件让 AI 参考文档
- 调整"Temperature"控制回答的创意程度

### 日志管理
- 定期导出重要的对话记录
- 清空不需要的旧日志以节省空间
- 使用导出的 JSON 数据进行后续分析

### 性能监控
- 查看"会话详情"中的执行时间
- 监控 Token 使用量
- 追踪技能调用的性能表现

## 🐛 故障排除

### 应用无法启动

```bash
# 检查虚拟环境
test -d .venv && echo "✅ 虚拟环境存在" || echo "❌ 需要创建虚拟环境"

# 检查依赖
arch -arm64 .venv/bin/python3.9 -m pip list | grep gradio

# 重新安装依赖
arch -arm64 .venv/bin/python3.9 -m pip install -r requirements.txt
```

### 日志无法保存

```bash
# 检查日志目录
test -d session_logs && echo "✅ 日志目录存在" || mkdir session_logs

# 检查权限
ls -ld session_logs
```

### 端口被占用

```bash
# 查看占用 7860 端口的进程
lsof -i :7860

# 查看占用 7861 端口的进程
lsof -i :7861

# 杀死进程（如需要）
kill -9 <PID>
```

## 📚 更多信息

- **详细文档**: 见 `SESSION_LOGGER_README.md`
- **实现细节**: 见 `IMPLEMENTATION_SUMMARY.md`
- **代码示例**: 见 `ui/session_logger.py` 和 `ui/session_viewer.py`

## ⚡ 快速命令

```bash
# 进入项目目录
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent

# 启动全部应用
./start_all.sh

# 单独启动 Chat Agent
arch -arm64 .venv/bin/python3.9 ui/web_agent_with_skills.py

# 单独启动 Session Viewer
arch -arm64 .venv/bin/python3.9 ui/session_viewer.py

# 查看所有日志
ls session_logs/

# 统计对话数
find session_logs -name "*.json" | wc -l

# 查看最新日志大小
ls -lSh session_logs/ | head -10
```

## 🎉 开始使用

现在您已经准备好了！

1. 运行 `./start_all.sh`
2. 打开 Chat Agent 进行对话
3. 打开 Session Viewer 查看日志
4. 享受完整的对话分析体验！

有任何问题，欢迎查阅详细文档或检查代码注释。

