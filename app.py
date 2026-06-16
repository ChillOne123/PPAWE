import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="PPAWE 学术撰写引擎", layout="wide")
st.title("🚀 PPAWE 极速学术撰写与自动引文引擎")

# ================= 1. API 与客户端配置 =================
# 优先从 .env 读取，如果没有，再回退到侧边栏输入
env_api_key = os.getenv("SILICONFLOW_API_KEY", "")
API_KEY = st.sidebar.text_input("硅基流动 API Key", value=env_api_key, type="password")

BASE_URL = "https://api.siliconflow.cn/v1" 
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Chat"

# 初始化客户端
if API_KEY:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
else:
    st.sidebar.warning("请先输入 API Key")

# ================= 2. 数据源加载与解析 =================
st.sidebar.header("文献元数据载入")
uploaded_file = st.sidebar.file_uploader("上传知网导出的 Excel/CSV (需包含：作者, 年份, 标题, 摘要)", type=["xlsx", "csv"])

references_context = ""
if uploaded_file is not None:
    # 兼容 csv 和 excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success(f"成功加载 {len(df)} 篇文献元数据！")
    
    # 将 DataFrame 转换为大模型易读的字符串格式
    # 假设你的列名大概是这些，如果不一样请在代码里微调
    # 核心逻辑：提取必需要素，摒弃冗余噪音
    for index, row in df.iterrows():
        # 容错处理，防止空值报错
        author = str(row.get('作者', '未知作者')).split(';')[0] # 取第一作者
        year = str(row.get('年份', '未知年份'))
        title = str(row.get('标题', '未知标题'))
        abstract = str(row.get('摘要', '无摘要'))[:200] # 截断超长摘要，节省 Token
        
        references_context += f"- 【文献ID】: ({author}, {year})\n  【标题】: {title}\n  【核心观点/摘要】: {abstract}\n\n"
        
    with st.expander("查看当前注入的文献知识库 (Prompt Context)"):
        st.text(references_context)

# ================= 3. 撰写控制台 =================
st.subheader("核心段落撰写")
draft_instruction = st.text_area(
    "输入撰写指令与大致结构 (例如：请论述公共政策执行中的街头官僚自由裁量权，分为三个层次...)", 
    height=150
)

if st.button("生成正文并自动插桩引文", type="primary"):
    if not API_KEY or not uploaded_file or not draft_instruction:
        st.error("请确保 API Key、文献文件和撰写指令都已提供！")
    else:
        with st.spinner("DeepSeek 正在结合文献撰写中..."):
            # 这里的 System Prompt 是压制幻觉、规范学术格式的灵魂
            system_prompt = """
            你是一个严谨的计算社会科学领域的学术研究员。你的任务是根据用户提供的【指令】撰写学术论文的正文段落。
            
            【绝对规则】：
            1. 你的论点必须并且只能基于下方提供的【文献知识库】。
            2. 严禁捏造、虚构任何学者、数据或理论（严禁幻觉）。
            3. 每当你引用、总结或借鉴了某篇文献的观点时，必须在句末严谨地插入引用标签，格式严格为：(作者, 年份)。
            4. 如果用户的指令超出了提供的文献库范围，请直接在正文中用括号标出：[此处缺乏相关文献支撑]。
            5. 使用专业、客观、学术化的书面表达。
            """
            
            user_prompt = f"【文献知识库】\n{references_context}\n\n【撰写指令】\n{draft_instruction}"
            
            try:
                # 调用模型
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3, # 调低温度，保证学术输出的稳定性和确定性
                    stream=True
                )
                
                # 流式输出结果，体验更好
                result_container = st.empty()
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        result_container.markdown(full_response)
                        
            except Exception as e:
                st.error(f"调用 API 失败: {e}")