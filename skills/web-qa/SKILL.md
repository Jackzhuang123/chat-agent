---
name: Web QA 测试
description: 使用浏览器自动化工具进行 Web 页面 QA 测试、用户流程验证和部署检查
tags: [web, qa, browser, testing, automation, playwright, selenium]
license: MIT
resources:
  - references/qa-checklist.md
---

# Web QA 测试技能

> 移植自 gstack SKILL.md 浏览器 QA 工作流，适配 chat-Agent 纯 Python 环境

## 概述

本技能提供系统化的 Web QA 测试方法论，包含：
- 用户流程验证（登录、注册、结账等）
- 部署检查和生产环境验证
- 响应式布局测试
- 表单验证测试
- 可访问性断言

## 工具选择

根据环境选择合适的浏览器自动化工具：

| 工具 | 适用场景 | 安装 |
|------|---------|------|
| Playwright | 现代 Web，推荐首选 | `pip install playwright && playwright install` |
| Selenium | 遗留项目兼容 | `pip install selenium` |
| requests + BeautifulSoup | 静态页面，无 JS | `pip install requests beautifulsoup4` |
| httpx | API 端点测试 | `pip install httpx` |

## QA 工作流

### 1. 用户流程测试（登录/注册/结账）

```python
from playwright.sync_api import sync_playwright

def test_login_flow(url: str, email: str, password: str):
    """测试登录流程 - 完整版（包含验证和截图证据）"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Step 1: 导航到页面
        page.goto(url)
        page.screenshot(path="/tmp/qa-before-login.png")

        # Step 2: 查看可交互元素（ARIA 辅助功能树）
        inputs = page.query_selector_all("input")
        print(f"发现 {len(inputs)} 个输入框")

        # Step 3: 填写表单
        page.fill('input[type="email"]', email)
        page.fill('input[type="password"]', password)

        # Step 4: 提交
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")

        # Step 5: 验证结果
        page.screenshot(path="/tmp/qa-after-login.png")
        assert ".dashboard" in page.content() or "欢迎" in page.title()

        browser.close()
        return "DONE: 登录流程验证成功"
```

### 2. 部署检查（生产环境验证）

```python
def check_deployment(url: str) -> dict:
    """部署健康检查 - 完整版（含控制台错误检测）"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # 收集控制台错误
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        # 收集网络失败
        failed_requests = []
        page.on("requestfailed", lambda req: failed_requests.append(req.url))

        page.goto(url, wait_until="networkidle")

        result = {
            "url": url,
            "title": page.title(),
            "status": "ok",
            "console_errors": console_errors,
            "failed_requests": failed_requests,
            "screenshot": "/tmp/qa-deployment-check.png",
        }

        page.screenshot(path=result["screenshot"])

        if console_errors or failed_requests:
            result["status"] = "DONE_WITH_CONCERNS"
            result["concerns"] = []
            if console_errors:
                result["concerns"].append(f"控制台错误 {len(console_errors)} 条: {console_errors[:3]}")
            if failed_requests:
                result["concerns"].append(f"网络失败请求 {len(failed_requests)} 条: {failed_requests[:3]}")
        else:
            result["status"] = "DONE"

        browser.close()
        return result
```

### 3. 响应式布局测试

```python
VIEWPORTS = [
    {"name": "mobile",  "width": 375,  "height": 812},   # iPhone
    {"name": "tablet",  "width": 768,  "height": 1024},  # iPad
    {"name": "desktop", "width": 1280, "height": 720},   # 标准桌面
]

def test_responsive(url: str, output_dir: str = "/tmp"):
    """响应式布局截图 - 三种视口"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        screenshots = []

        for viewport in VIEWPORTS:
            context = browser.new_context(viewport=viewport)
            page = context.new_page()
            page.goto(url)
            path = f"{output_dir}/qa-{viewport['name']}.png"
            page.screenshot(path=path)
            screenshots.append({"viewport": viewport["name"], "path": path})
            context.close()

        browser.close()
        return screenshots
```

### 4. 表单验证测试

```python
def test_form_validation(url: str):
    """测试表单验证 - 空提交 + 有效提交"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        # 空提交 - 检查验证错误出现
        page.click('button[type="submit"]')
        page.wait_for_timeout(500)

        errors = page.query_selector_all(".error, .error-message, [role='alert']")
        assert len(errors) > 0, "空提交应该显示验证错误"
        page.screenshot(path="/tmp/qa-form-empty.png")

        # 有效提交
        for input_el in page.query_selector_all("input[required]"):
            input_type = input_el.get_attribute("type") or "text"
            if input_type == "email":
                input_el.fill("test@example.com")
            elif input_type == "password":
                input_el.fill("Password123!")
            else:
                input_el.fill("测试输入")

        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.screenshot(path="/tmp/qa-form-valid.png")

        browser.close()
        return "DONE: 表单验证测试完成"
```

## 快速断言模式

```python
# 元素可见性
assert page.is_visible(".modal"), "模态框应可见"

# 按钮状态
assert page.is_enabled("#submit-btn"), "提交按钮应启用"

# 页面包含文本
assert "Success" in page.content()

# 元素数量
items = page.query_selector_all(".list-item")
assert len(items) > 0, "列表不应为空"

# 标题检查
assert page.title() == "预期标题"

# URL 检查
assert page.url.endswith("/dashboard")
```

## 完整性原则应用（Boil the Lake）

QA 测试最容易被省略，但 AI 时代它是最容易"煮沸的湖"。

| 常见捷径 | 完整方案 | 额外成本 |
|---------|---------|--------|
| 只测主路径 | 主路径 + 错误路径 + 边界 | +5分钟 |
| 只截最终截图 | 每步截图作为证据 | +2分钟 |
| 跳过移动端 | 三种视口全覆盖 | +3分钟 |
| 不检查控制台 | 捕获所有 JS 错误 | +1分钟 |

**始终选择完整方案**——AI 辅助下这些额外时间趋近于零。

## 完成状态报告

完成 QA 测试后，使用标准状态汇报：

```
STATUS: DONE
证据:
- 截图已保存至 /tmp/qa-*.png
- 控制台错误: 0
- 网络失败: 0
- 登录流程: ✅
- 响应式: ✅（375/768/1280px）
```

遇到问题时：
```
STATUS: DONE_WITH_CONCERNS
关注点:
1. 移动端 375px 布局有溢出（见 /tmp/qa-mobile.png）
2. 控制台有 2 条弃用警告（非阻塞）
```

## 注意事项

1. **截图作为证据**：每个关键步骤都应截图，不能只说"测试通过"
2. **状态持久化**：使用 `browser_context` 保持 Cookie 跨页面持久化
3. **网络空闲等待**：用 `wait_for_load_state("networkidle")` 而非固定等待
4. **隐式断言**：用 `assert` 替代 print，失败立即暴露
5. **并行安全**：测试间不共享浏览器实例

