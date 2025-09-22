import tiktoken
import os

USE_TIKTOKEN = os.getenv("USE_TIKTOKEN", "false").lower() == "true"

def approx_token_count(s: str) -> int:
    return max(1, len(s) / 4)

def tok_count(s: str) -> int:
    if USE_TIKTOKEN:
        try:
            tokenc = tiktoken.get_encoding("cl100k_base")
            return len(tokenc.encode(s))
        except Exception:
            pass
    return approx_token_count(s)