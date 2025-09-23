from google import genai
from google.genai import types
from utils.jinja_template import render_template
import logging
import dotenv

dotenv.load_dotenv()
client = genai.Client()
logging.getLogger("gemini")

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
    logging.info("Total tokens: %d", resp.usage_metadata.total_token_count)
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

    logging.info("Total tokens: %d", resp.usage_metadata.total_token_count)

    if hasattr(resp, "thinking") and resp.thinking:
        logging.info("Model reasoning trace:\n%s", resp.thinking)

    return resp.text
