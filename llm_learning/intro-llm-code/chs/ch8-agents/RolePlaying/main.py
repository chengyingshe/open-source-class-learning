'''2.4.1 手工编写代码-角色扮演'''

from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.models import ModelFactory


model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key='YOUR_API_KEY'
)

role_play_session = RolePlaying(                                # 直接调用核心类
    assistant_role_name="Python Programmer",                    # 指定助手智能体的具体身份
    assistant_agent_kwargs=dict(model=model),                   # 传递助手智能体的相关参数
    user_role_name="Stock Trader",                              # 指定用户智能体的具体身份
    user_agent_kwargs=dict(model=model),                        # 传递用户智能体的相关参数
    task_prompt="Develop a trading bot for the stock market",   # 给定初始任务提示
    with_task_specify=True,                                     # 选择是否需要进一步明确任务   
    task_specify_agent_kwargs=dict(model=model)                 # 传递任务明确代理的相关参数
)


n=0
chat_turn_limit=50
while n < chat_turn_limit:  # 迭代轮次限制
    input_assistant_msg = model.init_chat()
    # 获取两个智能体的新一轮输出
    assistant_response, user_response = role_play_session.step(input_assistant_msg)

    # 判断两个智能体是否结束对话
    if assistant_response.terminated:
        print(f"AI Assistant terminated. Reason: {assistant_response.info['termination_reasons']}.")
        break
    if user_response.terminated:
        print(f"AI User terminated. Reason: {user_response.info['termination_reasons']}.")
        break

    # 打印角色扮演的对话内容
    print(f"AI User:\n{user_response.msg.content}\n")
    print(f"AI Assistant:\n{assistant_response.msg.content}\n")

    # 根据用户智能体的反馈判断任务是否完成
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        break

    input_assistant_msg = assistant_response.msg  # 更新角色扮演的下一轮输入
    n += 1  # 进行下一轮迭代
