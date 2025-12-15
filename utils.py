# utils.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import cohere

load_dotenv()

# Load embedding model (cached in memory)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Cohere Chat client
COHERE_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment. Add it to .env or env vars.")
co = cohere.Client(COHERE_KEY)

# ---------------- Embedding & similarity ----------------
def get_embedding(text: str):
    """Return a list embedding for the text."""
    return model.encode(text).tolist()

def cosine_similarity(a, b):
    """Cosine similarity between two vectors (lists)."""
    a = np.array(a)
    b = np.array(b)
    # guard against zero vectors
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------- Helper to extract text from Cohere response ----------------
def _extract_cohere_text(resp):
    """
    Cohere SDK returns different shapes depending on version.
    Try common attributes in order and return the first readable string.
    """
    # 1) message attribute (often present)
    msg = getattr(resp, "message", None)
    if msg:
        # if it's a string
        if isinstance(msg, str):
            return msg
        # if object with content or text
        content = getattr(msg, "content", None) or getattr(msg, "text", None)
        if content:
            return content if isinstance(content, str) else str(content)

    # 2) output_text (some versions)
    out_text = getattr(resp, "output_text", None)
    if out_text:
        return out_text

    # 3) text attribute (older generate responses)
    text_attr = getattr(resp, "text", None)
    if text_attr:
        return text_attr

    # 4) output or outputs (list or object)
    out = getattr(resp, "output", None) or getattr(resp, "outputs", None)
    if out:
        try:
            # if list-like
            if isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                # try common subfields
                for field in ("content", "text", "message", "output_text"):
                    v = first.get(field) if isinstance(first, dict) else getattr(first, field, None)
                    if v:
                        return v if isinstance(v, str) else str(v)
                return str(first)
            # if object
            return str(out)
        except Exception:
            return str(out)

    # 5) to_dict() if available
    to_dict = getattr(resp, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            return str(d)
        except Exception:
            pass

    # 6) fallback to string representation
    return str(resp)

# ---------------- Smart Chatbot using Cohere Chat API ----------------
def call_cohere_chat(transactions, user_question: str):
    """
    Use Cohere Chat API to answer finance questions.
    Builds a single-string prompt (system + user) and returns the assistant text.
    """
    # Build transaction text (include amount & category)
    transactions_text = "\n".join([
        f"{t.get('description','(no desc)')}: ${t.get('amount',0)} (Category: {t.get('category','general')})"
        for t in transactions
    ])

    # System + user prompt (single string)
    prompt = (
        "You are a helpful personal finance assistant. "
        "Given the user's transactions, always provide:\n"
        "- Total spent per category\n"
        "- A short overall summary\n"
        "- Friendly saving tips if applicable\n\n"
        f"User question: {user_question}\n\nTransactions:\n{transactions_text}"
    )

    try:
        resp = co.chat(
            model="command-nightly",
            message=prompt,
            max_tokens=300
        )
    except Exception as e:
        # Return error text so API responds instead of 500; you can also raise if you prefer
        return f"Error calling Cohere Chat API: {e}"

    # Extract text robustly
    answer = _extract_cohere_text(resp)
    return answer
