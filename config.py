import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Local / Hugging Face Models
WHISPER_MODEL = "openai/whisper-small"  # Runs fast on T4 GPU or local CPU
OPENAI_MODEL = "gpt-4o-mini"

# Target Settings
MAX_TOKENS = 150
DEFAULT_TEMPERATURE = 0