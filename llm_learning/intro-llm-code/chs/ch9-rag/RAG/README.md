# RAG实践

## 简介
本项目实现了基于 **Retrieval-Augmented Generation (RAG)** 的查询系统，包含了两种主要功能：
1. **基础 RAG 系统**：通过从文本数据中构建索引，进行检索并生成答案。
2. **查询分解与检索结果融合 RAG 系统**：基于查询分解技术，将一个输入查询分解成多个子查询，结合多个查询结果，进行融合优化后生成答案。

这两个系统都是基于 **LangChain** 框架，并结合了 **Ollama** 部署本地大语言模型进行推理和生成。

## 使用说明
- 首先，请确保已经安装了 **Ollama** 框架。
```
curl -fsSL https://ollama.com/install.sh | sh
```
- 然后，启动 **Ollama** 服务并运行 **Qwen** 模型。
```
ollama serve
ollama run qwen2.5
```
- 为RAG系统准备数据，并放至 `data/RAGDoc` 目录下。
- 运行基础RAG系统，测试查询问题。
```
python base_rag.py
```
- 运行查询分解与检索结果融合RAG系统，测试查询问题。
```
python multi_query_rag.py
```