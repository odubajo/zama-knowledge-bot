import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMADB_PATH = "./zama_comprehensive_db"

COLLECTION_NAME = "zama_content"

N_RESULTS = 6

PAGE_TITLE = "ZAMA educative community chatbot"
PAGE_ICON = "./zama.jpg"