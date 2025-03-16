'''2.6.2 查询分解与检索结果融合 RAG 系统'''

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
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from llama_index.core import SimpleDirectoryReader
from operator import itemgetter


#### 索引 ####
docs = SimpleDirectoryReader("data/RAGDoc").load_data()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()

# 使用 Ollama 接入本地大语言模型
llm = OllamaLLM(model="qwen2.5")

# 构造query分解Prompt
template = """You are a helpful assistant that generates multiple search queries based on 
              a single input query. \n
              Generate multiple search queries related to: {question} \n
              Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

#构造query分解链
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# 定义多查询融合函数
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # 初始化一个字典，用于存储每个文档的融合分数
    fused_scores = {}

    # 遍历每个文档
    for docs in results:
        # 根据排名遍历列表中的文档
        for rank, doc in enumerate(docs):
            # 将文档转换为字符串格式，作为键使用（假设文档可以序列化为JSON）
            doc_str = dumps(doc)
            # 如果文档尚未在融合分数字典fused_scores 中，则添加它，初始分数为0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 如果文档已存在，则检索其当前分数
            previous_score = fused_scores[doc_str]
            # 使用 RRF : 1 / (rank + k)公式 更新文档分数
            fused_scores[doc_str] += 1 / (rank + k)

    # 根据融合分数对文档进行排序，以获取最终的重排序结果
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # 将重排序结果作为包含文档和融合分数的元组列表返回
    return reranked_results

question = "复旦大学有几个校区?"

# 构建查询融合链
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
print(len(docs))

# 构建包含查询分解的 RAG 链
template = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"question": question}))
