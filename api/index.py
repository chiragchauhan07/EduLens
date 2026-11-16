"""
api/index.py — Vercel Serverless Entrypoint for EduLens Mini.

Routes incoming serverless requests to the streamlined EduLens Mini application.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
