import os
from urllib.parse import urljoin
from dotenv import load_dotenv
import httpx
from rich import print

load_dotenv()

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/"


def list_openai_models() -> list[dict]:
    # OpenAI /v1/models 的响应 schema 没有 pricing/free 字段,只能返回全量模型列表,
    # 由调用方按 id 判断哪些属于"免费"。base_url 与 api_key 从 .env 读取。
    resp = httpx.get(
        urljoin(base_url, "models"),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]


if __name__ == "__main__":
    print(list_openai_models())
