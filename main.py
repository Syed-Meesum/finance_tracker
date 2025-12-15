from fastapi import FastAPI
from models import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from utils import get_embedding, cosine_similarity, call_cohere_chat

app = FastAPI(title="Finance Tracker AI Backend")

# ---------------- Chat Endpoint ----------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest):
    answer = call_cohere_chat(body.transactions, body.question)
    return {"answer": answer}

# ---------------- Hybrid Search ----------------
@app.post("/hybrid-search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest):
    query_lower = body.query.lower()
    results = []

    # 1️⃣ Highest / Lowest logic
    if any(word in query_lower for word in ["highest", "most", "lowest", "least"]):
        if "highest" in query_lower or "most" in query_lower:
            tx = max(body.transactions, key=lambda t: t["amount"])
        else:
            tx = min(body.transactions, key=lambda t: t["amount"])

        return {
            "results": [{
                "id": "extreme",
                "text": f"{tx['description']}: ${tx['amount']} (Category: {tx.get('category','general')})",
                "score": 1.0
            }]
        }

    # 2️⃣ Query embedding
    query_embedding = get_embedding(body.query)

    # 3️⃣ Hybrid scoring
    for idx, t in enumerate(body.transactions):
        desc = t.get("description", "")
        category = t.get("category", "")
        text = f"{desc}: ${t.get('amount', 0)} (Category: {category})"

        text_embedding = get_embedding(text)
        semantic_score = cosine_similarity(query_embedding, text_embedding)

        keyword_score = 0.0
        for word in body.query.lower().split():
            if word in desc.lower():
                keyword_score += 0.7
            if word in category.lower():
                keyword_score += 0.3

        total_score = semantic_score + keyword_score

        if total_score > 0.3:
            results.append({
                "id": str(idx),
                "text": text,
                "score": round(total_score, 3)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:5]}
