'''2.4.2 LangChain 框架-链'''

from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 这是一个LLMChain，根据一部剧目的标题来撰写简介
llm = OpenAI(temperature=.7, openai_api_key='YOUR_API_KEY')
template = """You are a playwright. Given the title of play, it is your
job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# 这是一个LLMChain，根据剧目简介来撰写评论
llm = OpenAI(temperature=.7, openai_api_key='YOUR_API_KEY')
template = """You are a play critic from the New York Times. Given the synopsis of play,
it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

# 这是总体链，按顺序运行这两个链
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

# 运行总体链
title = "The Tragedy of Macbeth"
result = overall_chain.run(title)
print(result)