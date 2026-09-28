"""
FastAPI backend for the ZenPharma course support assistant.

Run from the project root:
    uvicorn app.main:app --reload --port 8000
"""
import os

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = "zenpharma_docs"
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4-large")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")   # cheaper option: claude-haiku-4-5-20251001
TOP_K = int(os.getenv("TOP_K", "5"))
# Cosine distance: lower = more similar. If the best match is worse than this,
# we skip Claude and send the student to the instructor. Tune with real questions.
MAX_DISTANCE = float(os.getenv("RELEVANCE_MAX_DISTANCE", "0.6"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

FALLBACK = (
    "I don't have this in the course material. Please post your question on the "
    "Udemy Q&A and the instructor will get back to you."
)

SYSTEM_PROMPT = """You are the support assistant for the ZenPharma DevOps course on Udemy.
Students ask about labs, errors and concepts from the course.

Rules:
- Answer ONLY using the provided context excerpts from the course documentation.
- If the context does not contain the answer, say so plainly and tell the student to post the question on Udemy Q&A. Never guess or invent commands, file names or settings.
- Be concise and practical: give the likely cause, then numbered fix steps. Put commands in code blocks.
- If the student pasted an error message, refer to it directly.
- Mention which document or section your answer comes from."""

app = FastAPI(title="ZenPharma Course Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

voyage = voyageai.Client(max_retries=3, timeout=30)
# Keys not scoped to a workspace must name one on every request.
WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")
claude = anthropic.Anthropic(
    default_headers={"anthropic-workspace-id": WORKSPACE_ID} if WORKSPACE_ID else None,
)
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(question) > 4000:
        raise HTTPException(status_code=400, detail="Question is too long (max 4000 characters)")

    # 1. Embed the question
    q_emb = voyage.embed([question], model=VOYAGE_MODEL, input_type="query").embeddings[0]

    # 2. Retrieve the closest chunks
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not docs or dists[0] > MAX_DISTANCE:
        return {"answer": FALLBACK, "sources": []}

    # 3. Ask Claude, grounded in the retrieved context
    context = "\n\n---\n\n".join(docs)
    message = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context from course documentation:\n\n{context}\n\nStudent question:\n{question}",
        }],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")

    sources = []
    for m in metas:
        label = f"{m['source']}#{m['section']}"
        if label not in sources:
            sources.append(label)

    return {"answer": answer, "sources": sources}