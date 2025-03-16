'''2.4.1 手工编写代码-辩论'''

import os
from openai import OpenAI


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user",
                "content": f"Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)

    return {"role": "user", "content": prefix_string}


agents = 3  # 指定参与的智能体个数
rounds = 2  # 指定迭代轮次上限
question = "Jimmy has $2 more than twice the money Ethel has. If Ethal has $8, how much money is Jimmy having?"  # 用户提出问题

agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)}] 
                   for agent in range(agents)]  # 为每一个智能体构造输入提示

client = OpenAI(
    api_key='YOUR_API_KEY'
)

for round in range(rounds):  # 对每一轮迭代
    for i, agent_context in enumerate(agent_contexts):  # 对每一个智能体
        if round != 0:  # 第一轮不存在来自其他智能体的发言
            # 获取除自己以外，其他智能体的发言
            agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
            # construct_message()函数：构造提示用作智能体的下一轮输入
            message = construct_message(agent_contexts_other, question, 2 * round - 1)
            agent_context.append(message)  # 将当前智能体的下一轮输入添加至列表

        completion = client.chat.completions.create(  # 进行发言
            model="gpt-3.5-turbo-0301",  # 选择模型
            messages=agent_context,  # 智能体的输入
            n=1
        )
        content = completion["choices"][0]["message"]["content"]  # 提取智能体生成的文本内容
        assistant_message = {"role": "assistant", "content": content}  # 修改角色为代理
        agent_context.append(assistant_message)  # 将当前智能体的发言添加至列表

        print(assistant_message['content'])
