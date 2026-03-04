import streamlit as st
import requests
import json
import time
from typing import List, Dict

# 页面配置
st.set_page_config(
    page_title="项目4：本地LLM聊天（Ollama）",
    page_icon="🦙",
    layout="wide",
)

st.title("🦙 本地LLM聊天（Ollama + gpt-oss:20b）")
#st.caption("项目驱动学习路径 - 项目4：本地运行开源大模型")

# ===============================
# 配置区（侧边栏）
# ===============================

with st.sidebar:
    st.header("⚙️ 配置")

    # 常用模型列表（你已经下载了 gpt-oss:20b）
    default_models = [
        "gpt-oss:20b",  # 你已经下载的模型
        "llama3.2:1b",
        "llama3.2:3b",
        "qwen2.5:3b",
    ]

    selected_model = st.selectbox(
        "选择模型",
        options=default_models,
        index=0,
        help="确保该模型已经通过 Ollama 下载（pull）",
    )

    custom_model = st.text_input(
        "或输入自定义模型名称（可选）",
        value="",
        help="例如：llama3.1:8b。如果填写，则优先使用自定义名称。",
    )

    # 实际使用的模型名
    model_name = custom_model.strip() or selected_model

    st.markdown("---")
    st.subheader("🎛️ 推理参数")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="控制回答的随机性，越高越有创造力，越低越稳健。",
    )

    max_tokens = st.slider(
        "最大生成长度 (num_predict)",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="一次回答最多生成的 token 数量。",
    )

    st.markdown("---")
    st.subheader("🧠 系统提示词")

    default_system_prompt = (
        "你是一个本地运行的中文 AI 助手，基于开源大模型 gpt-oss:20b。"
        "请使用清晰、友好的语气回答用户的问题，必要时可以分步骤说明。"
    )

    system_prompt = st.text_area(
        "System Prompt（可选）",
        value=default_system_prompt,
        height=120,
    )

    st.markdown("---")
    st.subheader("📊 状态")
    st.write(f"当前模型：`{model_name}`")
    st.info("确保已在终端或 Ollama GUI 中下载并运行该模型，例如：`ollama run gpt-oss:20b`")


# ===============================
# 会话状态
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []  # type: ignore[assignment]

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


# ===============================
# 调用 Ollama 的辅助函数（支持流式输出）
# ===============================

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"


def stream_ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """
    调用 Ollama 的 /api/chat 接口，流式返回内容片段。

    每次 yield 当前已生成的完整文本，用于在前端实时刷新。
    """

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL_CHAT, json=payload, stream=True, timeout=300)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "无法连接到 Ollama。请确认 Ollama 已启动，并在本机运行 `ollama serve` "
            "或打开 Ollama 桌面应用。"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("请求 Ollama 超时，请稍后重试或减小 max_tokens。")

    if resp.status_code != 200:
        # 尝试从返回值中解析错误信息
        try:
            data = resp.json()
            error_msg = data.get("error") or str(data)
        except Exception:
            error_msg = resp.text
        raise RuntimeError(f"Ollama 调用失败（{resp.status_code}）：{error_msg}")

    full_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except Exception:
            continue

        # /api/chat 的流式返回会多次给出 message.content 片段
        if "message" in data and isinstance(data["message"], dict):
            chunk = data["message"].get("content") or ""
            if chunk:
                full_text += chunk
                yield full_text

        if data.get("done"):
            break

    if not full_text:
        raise RuntimeError("Ollama 返回了空响应，请稍后重试。")


# ===============================
# 主聊天区域
# ===============================

prompt = st.chat_input("输入消息，与本地大模型对话…")

if prompt:
    # 先显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构造发送给 Ollama 的历史消息
    messages_for_ollama: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages_for_ollama.append({"role": "system", "content": system_prompt.strip()})

    # 只取最近 10 条对话，避免太长
    recent_history = st.session_state.messages[-10:]
    for m in recent_history:
        messages_for_ollama.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.caption("💭 模型思考中，请稍候……")
        start_time = time.time()

        try:
            final_reply = ""
            for partial in stream_ollama_chat(
                model=model_name,
                messages=messages_for_ollama,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                final_reply = partial
                elapsed = time.time() - start_time
                # 实时更新内容和已用时间
                placeholder.markdown(final_reply + "▌")
                status.markdown(f"💭 模型思考中…… 已用时 {elapsed:.1f} 秒")

            elapsed = time.time() - start_time
            placeholder.markdown(final_reply)
            status.markdown(f"✅ 回答完成，用时 {elapsed:.1f} 秒")
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
        except Exception as e:
            status.markdown("")
            placeholder.error(str(e))


# 清除对话按钮
col_clear1, col_clear2 = st.columns(2)
with col_clear1:
    if st.button("🧹 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
#with col_clear2:
#    st.caption("本项目演示如何通过 HTTP 调用 Ollama 的本地大模型接口。")
