"""
main.py — EduLens FastAPI application.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlencode

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


class VercelPathNormalizerMiddleware:
    """
    Restore the original API route after Vercel rewrites /api/* to /api/index.py.
    vercel.json passes the original path in the __edulens_path query parameter.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            query_string = scope.get("query_string", b"")
            params = parse_qs(query_string.decode("latin-1"), keep_blank_values=True)
            original_paths = params.pop("__edulens_path", [])
            if path in ("/api/index.py", "/api/index", "/api", "/api/") and original_paths:
                original_path = original_paths[0]
                if original_path.startswith("/api/") or original_path in ("/health", "/model-info", "/sample-conversation", "/analyze"):
                    scope["path"] = original_path
                scope["query_string"] = urlencode(params, doseq=True).encode("latin-1")
        await self.app(scope, receive, send)


app.add_middleware(VercelPathNormalizerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The public API is available under /api/*; root aliases support local use and
# Vercel's normalized route handling without duplicating business logic.
app.include_router(analysis_router, prefix="/api")
app.include_router(analysis_router)

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "EduLens API is running. Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
