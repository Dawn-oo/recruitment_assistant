import os
from dotenv import load_dotenv


_ = load_dotenv()


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY 未配置"
    )