import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PPAWE 学术对话引擎", layout="wide")
st.title("🚀 PPAWE 2.0：对话式学术增强引擎")

# ================= 1. API 配置 =================
env_api_key = os.getenv("SILICONFLOW_API_KEY", "")
API_KEY = st.sidebar.text_input("硅基流动 API Key", value=env_api_key, type="password")
BASE_URL = "https://api.siliconflow.cn/v1" 
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro" 

if not API_KEY:
    st.sidebar.warning("请先输入 API Key")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 文献库无声注入 =================
st.sidebar.header("文献元数据注入区")
uploaded_file = st.sidebar.file_uploader("上传知网导出的 Excel/CSV", type=["xlsx", "csv"])

references_context = ""
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success(f"成功挂载 {len(df)} 篇文献！引擎已具备该领域的防幻觉装甲。")
    
    for index, row in df.iterrows():
        author = str(row.get('作者', '未知')).split(';')[0]
        year = str(row.get('年份', '未知'))
        title = str(row.get('标题', '未知'))
        abstract = str(row.get('摘要', '无'))[:500]
        references_context += f"- ({author}, {year}) | 标题: {title} | 摘要: {abstract}\n"

# ================= 3. 记忆与对话构建 =================
# 初始化对话历史记忆
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户新输入
if prompt := st.chat_input("和 PPAWE 探讨理论框架，或让它直接开始撰写正文..."):
    # 将用户输入存入记忆并显示在界面上
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 动态构建具有混合感知能力的 System Prompt
    system_prompt = f"""
    你是一个深谙公共管理理论与计算社会科学的顶尖学术助理。你不仅拥有宏大的理论视野，还具备极其严谨的治学态度。

    【工作模式】：
    你可以正常与用户进行头脑风暴、探讨理论（如多源流、协同治理、街头官僚等）、构建论文大纲。这部分你可以充分调用你的内在专业知识。

    【红线规则：引文防幻觉】：
    1. 当用户要求你**撰写学术论文的正文/文献综述**时，如果在论述中需要佐证、列举学者观点或研究结论，**必须且只能**从下方的【当前挂载文献库】中寻找依据，并在句末严格插入格式为 `(作者, 年份)` 的标注。
    2. 如果你想表达一个重要观点，但【当前挂载文献库】中找不到任何能支撑该观点的文献，你可以用自己的理论常识写出这段话，但在该段落结尾用方括号提示用户：[提示：此段逻辑缺乏实证文献支撑，建议补充相关数据或文献]。
    3. **绝对禁止**凭空捏造任何带有 (作者, 年份) 格式的引文。

    【当前挂载文献库】：
    {references_context if references_context else "当前未挂载任何文献。你可以自由交流，但在被要求撰写带引文的学术文本时，请提醒用户挂载数据。"}
    """

    # 组装发给 API 的消息（包含系统指令和过去的记忆）
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # 模型输出响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.5, # 稍微调高温度，让它的理论推演更灵活，但系统提示词会锁死引文格式
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"引擎响应异常: {e}")
            full_response = "出错了，请检查 API 或网络连接。"

    # 将助手的回答存入记忆
    st.session_state.messages.append({"role": "assistant", "content": full_response})