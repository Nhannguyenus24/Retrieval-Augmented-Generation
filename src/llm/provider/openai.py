from openai import OpenAI
from utils.jinja_template import render_template

client = OpenAI()

def one_shot(
    contents: str,
    user_query: str,
    template_name: str = "rag_qa.jinja",
    model: str = "gpt-4o-mini",
    is_format: bool = False,
) -> str:
    """
    Generate response using OpenAI with Jinja2 template
    """
    prompt = render_template(
        template_name=template_name,
        contents=contents,
        user_query=user_query
    )

    if not prompt:
        return "Error: Could not render prompt template"

    # Build params
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    if is_format:
        params["text_format"] = "markdown"

    resp = client.chat.completions.create(**params)

    if hasattr(resp, "usage") and resp.usage:
        print("Prompt tokens:", resp.usage.prompt_tokens)
        print("Completion tokens:", resp.usage.completion_tokens)
        print("Total tokens:", resp.usage.total_tokens)

    return resp.choices[0].message.content
