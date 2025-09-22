from typing import Dict, Optional
import os
from jinja2 import Environment, FileSystemLoader, Template
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


client = genai.Client()

# Initialize Jinja2 environment
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompt")
jinja_env = Environment(loader=FileSystemLoader(PROMPT_DIR))

def render_template(template_name: str, **kwargs) -> str:
    """
    Render a Jinja2 template with given parameters
    
    Args:
        template_name: Name of the template file (e.g., "rag_qa.jinja")
        **kwargs: Template variables
        
    Returns:
        str: Rendered template string
    """
    try:
        template = jinja_env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        return ""

def one_shot(contents: str, user_query: str, template_name: str = "rag_qa.jinja") -> str:
    """
    Generate response using Gemini with Jinja2 template
    
    Args:
        contents: Context content from retrieved chunks
        user_query: User's question
        template_name: Jinja template file to use
        
    Returns:
        str: Generated response
    """
    # Render prompt from template
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

def chat_with_context(user_query: str, context: Optional[str] = None, 
                     notebook: Optional[str] = None, template_name: str = "chat.jinja") -> str:
    """
    Generate response using chat template with context and notebook info
    
    Args:
        user_query: User's question
        context: Retrieved context/documents
        notebook: Project notebook information
        template_name: Jinja template file to use
        
    Returns:
        str: Generated response
    """
    # Render prompt from chat template
    prompt = render_template(
        template_name=template_name,
        user_query=user_query,
        context=context,
        notebook=notebook
    )
    
    if not prompt:
        return "Error: Could not render chat template"

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

def generate_with_template(template_name: str, model: str = "gemini-2.5-flash", **template_vars) -> str:
    """
    Generic function to generate content using any Jinja2 template
    
    Args:
        template_name: Name of the template file
        model: Gemini model to use
        **template_vars: Variables to pass to template
        
    Returns:
        str: Generated response
    """
    # Render prompt from template
    prompt = render_template(template_name=template_name, **template_vars)
    
    if not prompt:
        return f"Error: Could not render template {template_name}"

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    print("Prompt tokens:", resp.usage_metadata.prompt_token_count)
    print("Output tokens:", resp.usage_metadata.candidates_token_count)
    print("Total tokens:", resp.usage_metadata.total_token_count)
    return resp.text
