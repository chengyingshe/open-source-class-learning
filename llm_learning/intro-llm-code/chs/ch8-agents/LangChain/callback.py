'''2.4.2 LangChain 框架-回调'''

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI(openai_api_key='YOUR_API_KEY')
prompt = PromptTemplate.from_template("1 + {number} = ")

# 构造函数回调
# 首先，在初始化链时显式设置 StdOutCallbackHandler
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.run(number=2)

# 使用详细模式标志。然后，使用 verbose 标志实现相同的结果
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
chain.run(number=2)

# 请求回调。最后，使用请求的 callbacks 实现相同的结果
chain = LLMChain(llm=llm, prompt=prompt)
chain.run(number=2, callbacks=[handler])
