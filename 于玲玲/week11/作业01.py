import asyncio
import os
import uuid

from agents import RawResponsesStreamEvent, TResponseInputItem, trace, set_default_openai_api, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

os.environ["OPENAI_API_KEY"] = "sk-685d5da74c2047dbb7d80c0e80fcb05d"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
from agents import Agent, Runner


# ===== 子Agent 1: 情感分类 =====
emotion_agent = Agent(
    model="qwen-max", # 模型代号
    name="Emotion", # 给agent的取得名字（推荐英文，写的有意义）
    instructions="Your name is emo,you are an emotion classification tool.Tell me who you are before answering the question." # 对话中的 开头 system message
)

# ===== 子Agent 2: 实体识别 =====

entity_agent = Agent(
    model="qwen-max", # 模型代号
    name="Entity", # 给agent的取得名字（推荐英文，写的有意义）
    instructions="Your name is ent,you are a named entity recognition tool.Tell me who you are before answering the question." # 对话中的 开头 system message
)

# ===== 主Agent： =====
# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[emotion_agent, entity_agent],
)

async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    msg = input("你好，我可以帮你进行文本情感分类和文本实体识别，请问有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        # print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())