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
PUBLIC_DIR = ROOT_DIR / "public"
STATIC_DIR = PUBLIC_DIR if PUBLIC_DIR.exists() else FRONTEND_DIR
ASSETS_DIR = STATIC_DIR / "assets"


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
    redirect_slashes=False,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VercelPathNormalizerMiddleware:
    """
    ASGI middleware that restores original request paths when Vercel rewrites
    incoming requests (such as /api/*) to the /api/index.py serverless entrypoint.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8")
            forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("utf-8")
            target = matched_path or forwarded_uri
            if (path in ("/api/index.py", "/api/index", "/api", "/api/") or path.startswith("/api/index.py")) and target:
                scope["path"] = target
        await self.app(scope, receive, send)


app.add_middleware(VercelPathNormalizerMiddleware)

# Include API router with /api prefix and root for robust Vercel serverless routing
app.include_router(analysis_router, prefix="/api")
app.include_router(analysis_router)

# Mount frontend assets
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")



@app.get("/api/index.py", include_in_schema=False)
def index_py_fallback():
    """Serves health response if /api/index.py is accessed directly."""
    registry = ModelRegistry.get_instance()
    return {
        "status": "healthy",
        "service": "EduLens",
        "version": "1.0.0",
        "model_loaded": registry.pipeline is not None,
    }


@app.get("/", include_in_schema=False)
def serve_index():
    """Serves the EduLens single-page application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "EduLens API is running. Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

