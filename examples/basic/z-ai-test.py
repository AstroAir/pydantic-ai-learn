from openai import OpenAI  # noqa: N999

client = OpenAI(
    api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF", base_url="https://open.bigmodel.cn/api/paas/v4/"
)

completion = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
        {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事"},
    ],
    top_p=0.7,
    temperature=0.9,
)

print(completion.choices[0].message.content)
