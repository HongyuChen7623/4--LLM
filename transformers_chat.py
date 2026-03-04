import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def load_model(model_name: str):
    """
    加载本地 Transformers 模型（第一次会从 Hugging Face 下载）。

    默认使用 GPT-2（英文），主要目的是：
    - 体验直接用 Transformers 调模型的流程
    - 理解 tokenizer / model / generate 这些核心概念
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def generate_text(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只返回新增部分，避免重复展示 prompt
    return text[len(prompt) :]


st.set_page_config(
    page_title="项目4：Transformers 本地小模型 Demo",
    page_icon="🤗",
    layout="wide",
)

st.title("🤗 本地小模型 Demo（Transformers + GPT-2）")
#st.caption("项目驱动学习路径 - 项目4：方案B 使用 Transformers 库（更专业）")

with st.sidebar:
    st.header("⚙️ 配置")
    model_name = st.selectbox(
        "选择模型（建议先从小模型体验）",
        ["gpt2", "gpt2-medium", "distilgpt2"],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    # GPT-2 系列的总上下文长度通常为 1024 tokens，这里给出 400 的上限以平衡速度与长度
    max_new_tokens = st.slider(
        "max_new_tokens（新增 token 数）",
        20,
        400,
        120,
        20,
        help="值越大，生成越长，但推理时间也越久（CPU 上建议控制在 400 以内）。",
    )

    st.markdown("---")
    st.info(
        "第一次运行会从 Hugging Face 下载模型权重，可能比较慢；"
        "后续离线也可以使用。"
    )

tokenizer, model, device = load_model(model_name)

st.subheader("📝 输入 Prompt")
prompt = st.text_area(
    "提示词（建议用英文效果更好，例如：'Write a short story about a dragon.'）",
    height=150,
)

if st.button("🚀 生成", type="primary"):
    if not prompt.strip():
        st.error("请输入提示词。")
    else:
        with st.spinner("模型生成中，请稍候……"):
            output = generate_text(
                tokenizer,
                model,
                device,
                prompt.strip(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        st.markdown("### ✨ 输出结果")
        st.write(output)


