from fastapi import FastAPI
from models import ChatRequest, ChatResponse, SearchRequest, SearchResponse, SearchResult
from utils import get_embedding, cosine_similarity
from utils import call_cohere_chat  # <- use this




# ---------------- Chatbot Endpoint ----------------


app = FastAPI(title="Finance Tracker AI Backend")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest):
    answer = call_cohere_chat(body.transactions, body.question)
    return {"answer": answer}


@app.post("/hybrid-search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest):
    query_lower = body.query.lower()
    results = []

    # 1️⃣ Handle "highest"/"lowest" queries
    if any(word in query_lower for word in ["highest", "most", "lowest", "least"]):
        if "highest" in query_lower or "most" in query_lower:
            max_tx = max(body.transactions, key=lambda t: t['amount'])
            results.append({
                "id": "max",
                "text": f"{max_tx['description']}: ${max_tx['amount']} (Category: {max_tx.get('category', 'general')})",
                "score": 1.0
            })
        elif "lowest" in query_lower or "least" in query_lower:
            min_tx = min(body.transactions, key=lambda t: t['amount'])
            results.append({
                "id": "min",
                "text": f"{min_tx['description']}: ${min_tx['amount']} (Category: {min_tx.get('category', 'general')})",
                "score": 1.0
            })
        return {"results": results}

    # 2️⃣ Semantic embedding of query
    query_embed = get_embedding(body.query)

    # 3️⃣ Hybrid scoring: semantic + keyword
    for idx, t in enumerate(body.transactions):
        desc = t.get("description", "")
        category = t.get("category", "")
        text = f"{desc}: ${t.get('amount', 0)} (Category: {category})"

        # Semantic similarity
        text_embed = get_embedding(text)
        sem_score = cosine_similarity(query_embed, text_embed)

        # Keyword match boost
        keyword_score = 0.0
        if any(word in desc.lower() for word in body.query.lower().split()):
            keyword_score += 0.7  # strong boost if keyword found in description
        if any(word in category.lower() for word in body.query.lower().split()):
            keyword_score += 0.3  # smaller boost for category match

        # Combined score
        total_score = sem_score + keyword_score
        if total_score > 0.3:  # filter irrelevant transactions
            results.append({
                "id": str(idx),
                "text": text,
                "score": round(total_score, 3)
            })

    # 4️⃣ Sort by combined score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # 5️⃣ Limit results for Flutter app
    return {"results": results[:5]}