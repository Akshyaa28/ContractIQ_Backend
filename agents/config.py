"""Agent layer configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TEMPERATURE = 0.3
GROQ_MAX_TOKENS = 2000
