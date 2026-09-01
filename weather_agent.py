import os
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from rich import print
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain.agents.middleware.types import AgentState, dynamic_prompt
from typing import Callable, TypedDict

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """你是一位擅长说双关语的天气预报专家。

你可以使用两种工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道地点。如果你能从问题中判断他们指的是他们所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

system_message = SystemMessage(
    content=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]
)

advanced_model = init_chat_model(
    model_provider="openai",
    model="gpt-4o-mini",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

basic_model = init_chat_model(
    model_provider="deepseek",
    model="deepseek-v4-flash",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)


@wrap_model_call
def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])
    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)
    return handler(request)


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str | None
    user_role: str | None


class AuthAgentState(AgentState):
    """扩展的 Agent state,新增 authenticated 字段供 state_based_tools 中间件读取。"""
    authenticated: bool


@dataclass
class ResponseFormat:
    """智能体的响应模式。"""
    # 一个双关语回复（始终必需）
    punny_response: str
    # 官方正式回复
    official_response: str
    # 如果有的话，关于天气的任何有趣信息
    weather_conditions: str | None = None


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 检索用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("__main__", "ResponseFormat"),
            ("__main__", "Context"),
        ]
    )
)

agent = create_agent(
    name="weather_agent",
    model=advanced_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    middleware=[state_based_tools, user_role_prompt, handle_tool_errors],
    state_schema=AuthAgentState,
    context_schema=Context,
    response_format=ProviderStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": HumanMessage("what is the weather outside?"),
        "authenticated": True,
    },
    config=config,
    context=Context(user_id="1", user_role="expert")
)

print(response)
