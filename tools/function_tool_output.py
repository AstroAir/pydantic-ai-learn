from datetime import datetime

from pydantic import BaseModel
from pydantic_ai import Agent, DocumentUrl, ImageUrl
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

agent = Agent(
    model,
)


class User(BaseModel):
    name: str
    age: int


@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()


@agent.tool_plain
def get_user() -> User:
    return User(name="John", age=30)


@agent.tool_plain
def get_company_logo() -> ImageUrl:
    return ImageUrl(url="https://iili.io/3Hs4FMg.png")


@agent.tool_plain
def get_document() -> DocumentUrl:
    return DocumentUrl(url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")


result = agent.run_sync("What time is it?")
print(result.output)
# > The current time is 10:45 PM on April 17, 2025.

result = agent.run_sync("What is the user name?")
print(result.output)
# > The user's name is John.

# result = agent.run_sync('What is the company name in the logo?')
# print(result.output)
# > The company name in the logo is "Pydantic."

result = agent.run_sync("What is the main content of the document?")
print(result.output)
# > The document contains just the text "Dummy PDF file."
