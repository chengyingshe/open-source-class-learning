# LangChain框架实践

## 简介
本项目展示了如何使用 **LangChain** 框架的不同功能，构建一些常见的应用场景，帮助开发者快速掌握 LangChain 的核心模块。展示了如何使用 LangChain 实现一个基于 **RAG** 的语言模型，并展示了如何使用 LangChain 的数据连接、链式操作和回调功能来实现应用。

1. **模型输入/输出**：LangChain 提供了处理模型输入和输出的能力，可以将不同的数据格式转换为适合语言模型的输入，同时也能将语言模型的输出解析为需要的格式。通过 LangChain，可以轻松地管理模型的输入和输出流，并进行适当的后处理。
2. **数据连接**：LangChain 提供了丰富的数据连接功能，使得开发者可以轻松地将外部数据源（如数据库、API、文件等）与语言模型结合，实现数据驱动的应用。
3. **链**：LangChain 的链式操作模块允许将多个操作组合在一起，形成一个数据处理流程。可以创建多个模块并通过管道连接它们，像“流水线”一样处理数据流。
4. **记忆**：LangChain 提供了“记忆”功能，允许模型保留对话历史或上下文。这对于构建对话系统或多轮交互式应用非常有用。
5. **回调**：LangChain 提供了回调机制，可以在链中各个步骤完成后触发自定义操作。回调可以帮助开发者更灵活地控制流程，比如记录日志、监控执行进度等。
6. **RAG实践**：展示了如何使用 LangChain 实现一个基于 RAG 的语言模型。

## 使用说明
- 从 [OpenAI](https://openai.com/api/) 申请获取 **api key** ，并替换各测试文件中的 `YOUR_API_KEY` 变量。RAG实践中，需要额外从 [SerpAPI](https://serpapi.com/manage-api-key) 申请 **serpapi api key** ，用于搜索引擎查询，并替换 `SERPAPI_API_KEY` 变量。

- **模型输入/输出** 功能测试。
```
python model_io.py
```
- **数据连接** 功能测试。
```
python data_connection.py
```
- **链** 功能测试。
```
python chain.py
```
- **记忆** 功能测试。
```
python memory.py
```
- **回调** 功能测试。
```
python callback.py
```
- **RAG** 实践。
```
python rag.py
```