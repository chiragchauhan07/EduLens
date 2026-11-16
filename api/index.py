"""
api/index.py — Vercel Serverless Entrypoint for EduLens.

Routes incoming serverless requests to the streamlined EduLens application.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
