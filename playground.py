from dataclasses import dataclass
from typing import TypedDict
from pydantic import BaseModel


class Context(BaseModel):
    """自定义运行时上下文模式。"""
    user_id: str | None
    user_role: str | None


context = Context(user_id="1", user_role="expert")

print(context.user_id)
