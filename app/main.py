from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic using the modern lifespan pattern."""
    # ── Startup ────────────────────────────────────────────────────────────
    print("🚀 Initializing RAG system…")
    from app.Services import vector_db
    vector_db.load_index()


    print("✅ RAG system ready. Embedding model loads on first query.")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    print("🛑 Shutting down RAG system.")


app = FastAPI(
    title="Smart RAG System",
    description="Multimodal document understanding + RAG-powered assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production (Render, etc.)
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.Routes import upload, query, compare, convert, docs, audio

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router,  prefix="/query",  tags=["Query"])
app.include_router(compare.router, prefix="/compare", tags=["Compare"])
app.include_router(convert.router, prefix="/convert", tags=["Convert"])
app.include_router(docs.router,   prefix="/docs",   tags=["Documents"])
app.include_router(audio.router,  prefix="/audio",  tags=["Audio"])

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    """Serve the chat frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(html_path)
