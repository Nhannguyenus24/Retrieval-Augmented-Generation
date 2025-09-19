from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how RAG works",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)
