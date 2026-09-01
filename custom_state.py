import json
from dataclasses import dataclass

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema: CustomState = CustomState


cs = CustomState()
cs["user_preferences"] = {"aa": 111}
print(cs)

cm = CustomMiddleware()
cm.state_schema = cs

print(json.dumps(cm, default=lambda o: getattr(o, "__dict__", str(o)), indent=2, ensure_ascii=False))
