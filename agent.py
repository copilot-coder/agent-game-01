import os
import json
import random
from itertools import permutations, product
from typing import List, Iterable, Union

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv("OPENAI_API_BASE")
model = os.getenv('MODEL') or "gpt-3.5-turbo"
stream = True  # 是否流式输出
debug = False  # 是否打印调试信息

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 系统角色
system_message = '''
你是一个24点游戏助手。
- 开始游戏时，你需要生成一组随机数，提示用户回答，然后校验用户的回答。
- 如果用户表示回答不了问题，你可以生成答案。
- 用户可以向你提供一组数字提问如何计算，你需要根据这组数字生成答案。
'''

tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_random_numbers",
            "description": "为24点游戏生成一组随机数",
            "parameters": {
                "type": "object",
                "properties": {
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_answer",
            "description": "校验表达式计算结果是否为24",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "表达式"
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_answer",
            "description": "根据一组随机数生成24点游戏的答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "一组随机数"
                    }
                },
                "required": [
                    "numbers"
                ]
            }
        }
    }
]


def generate_random_numbers():
    while True:
        numbers = []
        while len(numbers) < 4:
            num = random.randint(1, 13)
            if num not in numbers:
                numbers.append(num)
        res = generate_answer(numbers)
        # 确保生成的随机数能计算出24
        if res['code'] == 'ok':
            return {'numbers': numbers}


def check_answer(expression: str):
    try:
        val = eval(expression)
        if val == 24:
            result = {'code': 'ok'}
        else:
            result = {'code': 'error', 'msg': f"表达式{expression}计算结果为{val}, 不是24"}
    except Exception as e:
        result = {'code': 'error', 'msg': f"计算出错。{e}"}
    return result

def generate_answer(numbers):
    if len(numbers) != 4:
        return {'code': 'error', 'msg': "随机数个数不正确"}
    
    operations = ['+', '-', '*', '/']
    for num_perm in permutations(numbers):
        for ops in product(operations, repeat=3):
            # 尝试所有不同的括号组合
            expressions = [
                f'(({num_perm[0]} {ops[0]} {num_perm[1]}) {ops[1]} {num_perm[2]}) {ops[2]} {num_perm[3]}',
                f'({num_perm[0]} {ops[0]} ({num_perm[1]} {ops[1]} {num_perm[2]})) {ops[2]} {num_perm[3]}',
                f'({num_perm[0]} {ops[0]} {num_perm[1]}) {ops[1]} ({num_perm[2]} {ops[2]} {num_perm[3]})',
                f'{num_perm[0]} {ops[0]} (({num_perm[1]} {ops[1]} {num_perm[2]}) {ops[2]} {num_perm[3]})',
                f'{num_perm[0]} {ops[0]} ({num_perm[1]} {ops[1]} ({num_perm[2]} {ops[2]} {num_perm[3]}))',
            ]
            for expr in expressions:
                try:
                    if eval(expr) == 24:
                        return {'code': 'ok', 'answer': expr}
                except ZeroDivisionError:
                    continue
    return {'code': 'error'}


def invoke_tool(tool_call: Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]) -> ChatCompletionToolMessageParam:
    result = ChatCompletionToolMessageParam(
        role="tool", tool_call_id=tool_call.id)
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if func_name == "generate_random_numbers":
        res = generate_random_numbers()
    elif func_name == "check_answer":
        res = check_answer(args['expression'])
    elif func_name == "generate_answer":
        res = generate_answer(args['numbers'])
    else:
        res = {'code': 'error', 'msg': '函数未定义'}
    result['content'] = str(res)
    return result


def merge_too_calls(tool_calls: List[ChoiceDeltaToolCall], delta_tool_calls: List[ChoiceDeltaToolCall]):
    for delta_tool_call in delta_tool_calls:
        index = delta_tool_call.index
        if len(tool_calls) <= index:
            if delta_tool_call.function.arguments is None:
                delta_tool_call.function.arguments = ''
            tool_calls.append(delta_tool_call)
            continue
        ref = tool_calls[index]
        ref.function.arguments += delta_tool_call.function.arguments


def main():
    MAX_MESSAGES_NUM = 40
    messages: Iterable[ChatCompletionMessageParam] = list()
    needInput = True
    while True:
        # 只保留MAX_MESSAGES_NUM条消息作为上下文
        if len(messages) > MAX_MESSAGES_NUM:
            messages = messages[-MAX_MESSAGES_NUM:]
            while len(messages) > 0:
                role = messages[0]['role']
                if role == 'system' or role == 'user':
                    break
                messages = messages[1:]

        # 等待用户输入
        if needInput:
            query = input("\n>>>> 请输入:").strip()
            if query == "":
                continue
            messages.append(ChatCompletionUserMessageParam(
                role="user", content=query))

        # 插入系统消息
        if messages[0]['role'] != 'system':
            messages.insert(0, ChatCompletionSystemMessageParam(
                role="system", content=system_message
            ))
        # 向LLM发起查询（除了用户的query外，还需要带上tools定义）
        chat_completion = client.chat.completions.create(
            messages=messages,
            tools=tools,
            model=model,
            stream=stream
        )

        tool_calls = None
        content = None
        if stream:
            # 处理流式输出
            for chunk in chat_completion:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if isinstance(delta.tool_calls, list):
                    if tool_calls is None:
                        tool_calls = []
                    merge_too_calls(tool_calls, delta.tool_calls)
                elif isinstance(delta.content, str):
                    if content is None:
                        content = ""
                    content += delta.content
                    print(delta.content, end='', flush=True)
        else:
            # 非流式输出
            tool_calls = chat_completion.choices[0].message.tool_calls
            content = chat_completion.choices[0].message.content
        if isinstance(tool_calls, list) and len(tool_calls) > 0:  # LLM的响应信息有tool_calls信息
            needInput = False
            messages.append(ChatCompletionAssistantMessageParam(
                role="assistant", tool_calls=tool_calls, content=''))
            # 注意：LLM的响应可能包括多个tool_call
            if debug:
                print('[debug] tool_calls:', tool_calls)
            for tool_call in tool_calls:
                result = invoke_tool(tool_call)
                if debug:
                    print('[debug] result:', result)
                messages.append(result)
        else:
            needInput = True
            if isinstance(content, str) and len(content) > 0:
                if not stream:
                    print(content)
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant", content=content))


main()
