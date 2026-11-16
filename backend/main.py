"""
main.py — EduLens FastAPI Application.

Serves the simplified machine learning inference API and static web frontend
for student understanding and expressed confidence analysis.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.analysis import router as analysis_router
from backend.services.prediction import ModelRegistry

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load ML model once during startup
    print("Initializing EduLens ML Model Registry...")
    registry = ModelRegistry.get_instance()
    print(f"Model loaded successfully: {registry.classes} ({len(registry.classes)} classes)")
    yield
    print("Shutting down EduLens.")


app = FastAPI(
    title="EduLens",
    description="Student Learning Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(analysis_router)

# Mount frontend assets
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
def serve_index():
    """Serves the EduLens single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "EduLens API is running. Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

