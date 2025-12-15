# utils.py
import os
from dotenv import load_dotenv
import cohere

load_dotenv()

# ---------------- Cohere Client ----------------
COHERE_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_KEY:
    raise RuntimeError("COHERE_API_KEY not found")

co = cohere.Client(COHERE_KEY)

# ---------------- Embeddings (Cohere API) ----------------
def get_embedding(text: str):
    """
    Get semantic embedding using Cohere (cloud-based, low memory).
    """
    response = co.embed(
        texts=[text],
        model="embed-english-light-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ---------------- Chatbot ----------------
def call_cohere_chat(transactions, user_question: str):
    transactions_text = "\n".join([
        f"{t.get('description','(no desc)')}: ${t.get('amount',0)} (Category: {t.get('category','general')})"
        for t in transactions
    ])

    prompt = f"""
You are a helpful personal finance assistant.

Always:
- Show total spent per category
- Give a short overall summary
- Provide friendly saving tips

User question:
{user_question}

Transactions:
{transactions_text}
"""

    response = co.chat(
        model="command-nightly",
        message=prompt,
        max_tokens=300
    )

    return response.text
