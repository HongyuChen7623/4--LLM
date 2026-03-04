# 🦙 项目4：本地LLM聊天（Ollama + gpt-oss:20b）

基于 **Ollama 本地大模型** 和 **Streamlit** 的聊天应用

本项目的目标：

* 体验完全在本地运行开源大模型的流程

* 学会通过 HTTP 接口调用 Ollama

* 一个「本地 LLM 部署 + 工程化界面」的可展示项目

## ✨ 功能特性

* **本地推理，无需云端 API**

* 支持多种模型（默认包含 `gpt-oss:20b`，也可输入自定义模型名）

* **多轮对话**：保留最近 10 轮历史，构造上下文

* **System Prompt 可配置**：轻松做角色扮演 / 专家助手

* 可调节：

  * `Temperature`（创造性）

  * `num_predict`（最大生成长度）

* 一键清除对话历史

## 🧱 项目结构

```text
项目4Ollama/
├── import requests.py   # 从学习路径复制的最小调用示例（命令行测试用）
├── ollama_chat.py       # ✅ 推荐使用的 Streamlit 网页聊天应用
└── requirements.txt     # Python 依赖
```

## 🚀 环境准备

### 1. 安装 Ollama

1. 前往 `https://ollama.com/download` 下载并安装（Windows 有安装包）

2. 安装完成后，启动 Ollama（桌面应用或命令行均可）

### 2. 下载模型

在命令行中执行：

```bash
ollama pull gpt-oss:20b
```

如果想尝试其他模型，例如：

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

## 🔧 安装依赖

在项目目录下执行：

```bash
cd 项目4Ollama
pip install -r requirements.txt
```

`requirements.txt` 内容：

```txt
streamlit>=1.28.0
requests>=2.31.0
```

## 🖥️ 运行应用

1. 确保 Ollama 已启动（看到托盘图标，或者命令行 `ollama serve` 正在运行）

2. 在命令行执行：

```bash
cd 项目4Ollama
streamlit run ollama_chat.py
```

3. 浏览器会自动打开 `http://localhost:8501`，看到标题：

> 🦙 本地LLM聊天（Ollama + gpt-oss:20b）

## 💡 使用说明

### 1. 选择模型

在左侧侧边栏：

* 下拉框：在 `gpt-oss:20b / llama3.2:1b / llama3.2:3b / qwen2.5:3b` 之间切换

* 文本框：可以输入任意已在 Ollama 中存在的模型名（如 `llama3.1:8b`），会优先使用你输入的名称

> 提示：如果使用自定义模型名，记得先在命令行 `ollama pull 模型名`。

### 2. 配置推理参数

* **Temperature**

  * 越低（如 0.1）输出更「保守、稳定」

  * 越高（如 1.0）输出更「发散、有创造力」

* **最大生成长度 (num_predict)**

  * 控制一次回答最长生成多少 token

### 3. 设置 System Prompt

在侧边栏可以编辑系统提示词，例如：

* 「你是一个耐心的 Python 老师，专门给零基础同学讲解编程。」

* 「你是一个数据分析专家，回答时优先给出分析步骤和结论。」

System Prompt 会作为对话的第一条消息发送给模型，影响整体风格。

### 4. 开始对话

在底部输入框输入问题，例如：

* 「你好，你是谁？」

* 「用通俗的例子解释一下什么是过拟合？」

* 「帮我设计一个 3 天的 Python 入门学习计划。」

模型会基于最近 10 轮对话历史生成回答，实现多轮上下文。

### 5. 清除对话

点击页面底部的「🧹 清除对话历史」按钮，可以清空当前会话，重新开始。

##

相比文档示例，本项目做了以下增强：

* 使用 `/api/chat` 接口支持多轮对话，而不是一次性 `generate`

* 增加 System Prompt 配置，便于做 Prompt 实验

* 加入基础错误处理（Ollama 未启动、超时、HTTP 错误等）

## 📝 简历描述参考

可以在简历中这样写本项目（示例）：

> **本地开源大模型聊天系统（Ollama + gpt-oss:20b）**
>
> * 技术栈：Ollama、开源大模型 gpt-oss:20b、Streamlit、Python、HTTP API
>
> * 功能：基于 Ollama 部署本地大模型，实现多轮对话、本地推理；支持模型切换、System Prompt 配置和推理参数调节
>
> * 亮点：完整打通「模型下载 → 本地服务 → Python 调用 → Web 界面」链路，对比云端 API，理解推理延迟与资源占用；为后续 RAG、本地 Agent 项目打下基础

## ❗ 常见问题

**Q1：页面报错“无法连接到 Ollama”？**

* 检查是否已启动 Ollama（桌面版或命令行 `ollama serve`）

* 确认本机能访问 `http://localhost:11434`

**Q2：提示模型不存在或加载失败？**

* 确认在命令行执行过 `ollama pull gpt-oss:20b` 或对应模型名

* 模型首次加载较慢，耐心等待几分钟

**Q3：回答很慢？**

* 尝试：

  * 换用更小的模型（如 `llama3.2:3b`）

  * 减小 `max_tokens`

  * 关闭其他占用 CPU / 内存较高的程序

准备好后，你可以在此基础上继续做「本地 RAG」「本地 Agent」等进阶项目，将本地模型和你前两个项目（Chatbot、RAG）打通。祝实验顺利！🦙
