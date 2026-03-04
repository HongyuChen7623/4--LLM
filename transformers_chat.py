import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def load_model(model_name: str):
    """
    加载 Transformers 模型。

    - 如果传入的是模型名（如 "gpt2"），则从 Hugging Face Hub 下载预训练权重；
    - 如果传入的是本地目录（如 "../项目5Finetune/my_finetuned_model"），
      则从该目录加载你在项目5中微调后的模型。
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
st.caption("既可以体验预训练 GPT-2，也可以加载项目5微调后的 GPT-2。")

# 检查项目5是否已生成微调模型目录
finetuned_dir = os.path.join("..", "项目5Finetune", "my_finetuned_model")
has_finetuned = os.path.isdir(finetuned_dir)

with st.sidebar:
    st.header("⚙️ 模型选择")

    # 根据是否存在微调模型，决定下拉选项
    if has_finetuned:
        model_choice = st.selectbox(
            "选择模型来源",
            ["预训练 GPT-2（gpt2）", "微调后 GPT-2（项目5Finetune/my_finetuned_model）"],
            index=0,
        )
    else:
        model_choice = "预训练 GPT-2（gpt2）"
        st.info("未检测到 `项目5Finetune/my_finetuned_model`，当前仅可使用预训练 GPT-2。")

    if model_choice.startswith("预训练"):
        model_name = "gpt2"
        st.success("当前使用：预训练 GPT-2")
    else:
        model_name = finetuned_dir
        st.success("当前使用：项目5 微调后的 GPT-2")

    st.markdown("---")
    st.header("🎛️ 生成参数")
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
        "预训练模型会从 Hugging Face Hub 下载权重；\n"
        "微调模型则从本地目录 `项目5Finetune/my_finetuned_model` 加载。"
    )

tokenizer, model, device = load_model(model_name)

st.subheader("📝 输入 Prompt")
prompt = st.text_area(
    "提示词（建议用英文；如果选择微调模型，可输入与训练数据类似的问句，"
    "例如：'Q: How should I answer an interview question? A:'）",
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


