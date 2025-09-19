from typing import Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


client = genai.Client()


def one_shot(contents: str, user_query: str) -> str:

    prompt = f"""You are a RAG assistant. Answer ONLY based on the CONTEXT below.
Rules:
- Write a concise answer, directly addressing the query.
- Every important statement must include a citation in the format [file_name p. pages]. Example: [spec.pdf p. 12-14]
- If the CONTEXT does not provide enough information, clearly state: "The provided documents do not contain enough information."

USER QUERY:
{user_query}

CONTEXT (Top-k chunks):
{contents}

OUTPUT FORMAT:
- clear answer, as long and relative as posible.
- Insert citations immediately after the relevant statement, using the format [file_name p. pages] - content.
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    print("Prompt tokens:", resp.usage_metadata.prompt_token_count)
    print("Output tokens:", resp.usage_metadata.candidates_token_count)
    print("Total tokens:", resp.usage_metadata.total_token_count)
    return resp.text
