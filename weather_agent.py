import os
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from typing import Any, Literal
from langchain.agents.middleware import after_agent, AgentState
from langchain.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import Item
from langgraph.types import Command
from rich import print
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call, \
    SummarizationMiddleware
from langchain.agents.middleware.types import AgentState, dynamic_prompt, AgentMiddleware, after_model, before_model
from typing import Callable, TypedDict, Any
from pydantic import BaseModel, Field
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import UsageMetadataCallbackHandler

load_dotenv()

DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

model_name = "deepseek-v4-flash"

SYSTEM_PROMPT = """你是一位擅长说双关语的天气预报专家。

你可以使用的工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置
- set_user_location: 更新用户位置

如果用户询问天气，请确保你知道地点。如果你能从问题中判断他们指的是他们所在的位置，请使用 get_user_location 工具来查找他们的位置，并使用 set_user_location 更新位置。"""

system_message = SystemMessage(
    content=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]
)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # 1 request every 10s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=10,  # Controls the maximum burst size.
)

store = InMemoryStore()

advanced_model = init_chat_model(
    model_provider="deepseek",
    model=model_name,
    temperature=0.5,
    timeout=30,
    max_tokens=1000,
    max_retries=2,  # Default; increase for unreliable networks
    rate_limiter=rate_limiter,
).bind(logprobs=True)


class ResponseFormat(BaseModel):
    """智能体的响应模式。"""
    # 一个双关语回复（始终必需）
    punny_response: str = Field(description="punny response")
    # 官方正式回复
    official_response: str = Field(description="official response")


model_with_structure = advanced_model.with_structured_output(ResponseFormat)


@wrap_model_call
def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    is_authenticated = state.get("user_preferences").get("authenticated", False)
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
        print(e)
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
def get_weather_for_location(city: str, runtime: ToolRuntime) -> str:
    """获取指定城市的天气。"""
    writer = runtime.stream_writer

    # 在工具执行时流式传输自定义更新
    writer(f"正在查找城市数据：{city}")  # stream_mode="custom",
    writer(f"已获取城市数据：{city}")  # stream_mode="custom",

    return f"It's always sunny in {city}!"


@tool
def set_user_location(runtime: ToolRuntime, new_location: str) -> Command:
    """在对话状态中设置用户位置。"""
    preferences = runtime.state.get("user_preferences", {})
    runtime.store.put(("users",), "preferences", preferences)
    return Command(update={"user_location": new_location})


@dataclass
class UserContext:
    """自定义运行时上下文模式。"""
    user_id: str | None
    user_role: str | None


@tool
def get_user_location(runtime: ToolRuntime[UserContext]) -> str:
    """根据用户 ID 检索用户信息。"""
    user_id = runtime.context.user_id
    item: Item = runtime.store.get(("user_location",), user_id)
    return item.value if item else "Florida"


checkpointer = InMemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("__main__", "ResponseFormat"),
            ("__main__", "UserContext"),
        ]
    )
)


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    MAX_MESSAGE_LENGTH: int = 3
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGE_LENGTH:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGE_LENGTH:] if len(messages) % 2 == 0 else messages[-(MAX_MESSAGE_LENGTH + 1):]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret", "退出", "结束", "stop", "cancel"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [get_user_location, get_weather_for_location, set_user_location]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        print(state)


class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]


@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    # Get the model response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use another model to evaluate safety
    safety_model = init_chat_model(model_provider="deepseek", model=model_name)
    model_with_tools = safety_model.bind_tools([ResponseSafety])
    result = model_with_tools.invoke(
        [
            {
                "role": "system",
                "content": "Evaluate this AI response as generally safe or unsafe."
            },
            {
                "role": "user",
                "content": f"AI response: {last_message.text}"
            }
        ]
    )
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None


agent = create_agent(
    name="weather_agent",
    model=advanced_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location, set_user_location],
    middleware=[state_based_tools, user_role_prompt, trim_messages, validate_response, handle_tool_errors,
                safety_guardrail,
                CustomMiddleware(),
                SummarizationMiddleware(
                    model=model_name,
                    trigger=("tokens", 4000),
                    keep=("messages", 1)
                )],
    state_schema=CustomState,
    context_schema=UserContext,
    # response_format=ToolStrategy(ResponseFormat), # deepseek-v4-flash thinking模型不能自定义ResponseFormat
    checkpointer=checkpointer,
    store=store
)

callback = UsageMetadataCallbackHandler()

# `thread_id` 是给定对话的唯一标识符。
config = {
    "run_name": "weather_forecast",  # Custom name for this run
    "tags": ["weather"],  # Tags for categorization
    "metadata": {"user_id": "123"},  # Custom metadata
    "configurable": {
        "thread_id": "1"
    },
    "callbacks": [callback]
}

# response = agent.invoke(
#     {
#         "messages": HumanMessage("what is the weather outside?"),
#         "user_preferences": {"device": "ios",
#                              "authenticated": True},
#     },
#     config=config,
#     context=UserContext(user_id="1", user_role="expert")
# )
# print(response)

for chunk in agent.stream(
        {
            "messages": HumanMessage("what is the weather outside?"),
            "user_preferences": {"device": "ios", "authenticated": True},
        },
        stream_mode=["values", "updates", "custom"],
        config=config,
        context=UserContext(user_id="1", user_role="expert"),
        version="v2", ):
    print(chunk, "[bold yellow]==========================[/bold yellow]", flush=True)
    # # Each chunk contains the full state at that point
    # latest_message = chunk["messages"][-1]
    # if latest_message.content:
    #     if isinstance(latest_message, HumanMessage):
    #         print(f"User: {latest_message.content}")
    #     elif isinstance(latest_message, AIMessage):
    #         print(f"Agent: {latest_message.content}")
    # elif latest_message.tool_calls:
    #     print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
