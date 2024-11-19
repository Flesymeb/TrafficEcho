import json

from openai import OpenAI

client = OpenAI(
    api_key="sk-6SPxtFoADCQXHRVo9kkxFBBwLCbpHiKyC5gqUgLekriuvPiU",
    base_url="https://api.moonshot.cn/v1",
)

system_prompt = """
你是月之暗面（Kimi）的智能客服，你负责回答用户提出的各种问题。请参考文档内容回复用户的问题，你的回答可以是文字、图片、链接，在一次回复中可以同时包含文字、图片、链接。
 
请使用如下 JSON 格式输出你的回复：
 
{
    "text": "文字信息",
    "image": "图片地址",
    "url": "链接地址"
}
 
注意，请将文字信息放置在 `text` 字段中，将图片以 `oss://` 开头的链接形式放在 `image` 字段中，将普通链接放置在 `url` 字段中。
"""

completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
        },
        {
            "role": "system",
            "content": system_prompt,
        },  # <-- 将附带输出格式的 system prompt 提交给 Kimi
        {"role": "user", "content": "你好，我叫李雷，1+1等于多少？"},
    ],
    temperature=0.3,
    response_format={
        "type": "json_object"
    },  # <-- 使用 response_format 参数指定输出格式为 json_object
)

# 由于我们设置了 JSON Mode，Kimi 大模型返回的 message.content 为序列化后的 JSON Object 字符串，
# 我们使用 json.loads 解析其内容，将其反序列化为 python 中的字典 dict。
content = json.loads(completion.choices[0].message.content)

# 解析文本内容
if "text" in content:
    # 为了演示，我们将内容打印出来；
    # 在真实的业务逻辑中，你可能需要调用发送文本消息的接口将生成的文本发送给用户。
    print("text:", content["text"])

# 解析图片内容
if "image" in content:
    # 为了演示，我们将内容打印出来；
    # 在真实的业务逻辑中，你可能需要先解析图片地址，下载图片后，调用发送图片消息
    # 的接口将图片发送给用户。
    print("image:", content["image"])

# 解析链接
if "url" in content:
    # 为了演示，我们将内容打印出来；
    # 在真实的业务逻辑中，你可能需要调用发送链接卡片的接口，将链接以卡片的形式发送给用户。
    print("url:", content["url"])
