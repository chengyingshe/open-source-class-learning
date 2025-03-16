'''2.6.1 基础 RAG 系统'''

# 导入需要的模块和类
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from llama_index.core import SimpleDirectoryReader

#### 索引 ####
# 1. 从指定目录中读取所有文件的数据 
# 使用目录读取器 SimpleDirectoryReader 加载数据
docs = SimpleDirectoryReader("data/RAGDoc/").load_data()

# 2. 文件分割，采用滑动窗口方法进行分块，分块大小为1000，块之间重叠为200
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. 文本嵌入表示模型初始化
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 4. 使用Chroma构建向量检索
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()

#### 检索 和 生成 ####
# 3. 构建Prompt模板，使用现有的rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")

# 4. 使用 Ollama 接入本地大语言模型
llm = OllamaLLM(model="qwen2.5")

# 5. 检索后优化
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. 构建RAG链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 使用Rag链进行查询
response = rag_chain.invoke("复旦大学有几个校区?")

# 8. 打印从查询引擎返回的响应
print(response)
