from google import genai
from google.genai import types
from utils.jinja_template import render_template
import logging

client = genai.Client()

def one_shot(contents: str, user_query: str, template_name: str = "rag_qa.jinja") -> str:
    # Generate response using Gemini with Jinja2 template

    prompt = render_template(
        template_name=template_name,
        contents=contents,
        user_query=user_query
    )
    
    if not prompt:
        return "Error: Could not render prompt template"

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

def thinking(contents: str, user_query: str, template_name: str = "rag_qa.jinja", thinking_budget: int = 0) -> str:
    """
    Generate response using Gemini with optional reasoning (thinking).
    """

    prompt = render_template(
        template_name=template_name,
        contents=contents,
        user_query=user_query
    )
        
    if not prompt:
        return "Error: Could not render prompt template"

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        ),
    )

    print("Prompt tokens:", resp.usage_metadata.prompt_token_count)
    print("Output tokens:", resp.usage_metadata.candidates_token_count)
    print("Total tokens:", resp.usage_metadata.total_token_count)

    if hasattr(resp, "thinking") and resp.thinking:
        print("Model reasoning trace:\n", resp.thinking)

    return resp.text
