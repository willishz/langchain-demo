import os
from dataclasses import dataclass
from functools import wraps

from dotenv import load_dotenv
from rich import print
from typing import Callable, TypedDict, Any
from pydantic import BaseModel, Field

load_dotenv()


class ResponseFormat(BaseModel):
    """智能体的响应模式。"""
    punny_response: str | None = Field(default=None, description="punny response")
    official_response: str | None = Field(default=None, description="official response")  # 官方正式回复


@dataclass
class RequestFormat:
    """智能体的响应模式。"""
    punny_response: str | None = Field(default=None, description="punny response")
    official_response: str | None = Field(default=None, description="official response")  # 官方正式回复


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("before", func.__name__)
        func(*args, **kwargs)
        print("after", func.__name__)

    return wrapper


@log
def print_it(a, b):
    print(a, b)


@log
def print_it2(a, b):
    print(a, b)


res = RequestFormat(punny_response="a joke", official_response=None)
req = ResponseFormat(punny_response="a joke", official_response=None)
req2 = ResponseFormat()
print(res)
print(req2)
print_it('x', 'y')
print_it2('x', 'y')
