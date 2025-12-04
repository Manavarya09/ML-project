"""
FastAPI Web Application for Phishing Detection.
Serves a monochrome, Pinterest-style interface and provides an inference endpoint.
"""

from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.inference import predict_email_bert
from src.config import config

app = FastAPI(title="Phishing Detector Web App")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_email(email_text: str = Form(...)) -> Dict[str, Any]:
    """
    Analyze email text using the BERT model.
    Returns JSON with prediction and explanation.
    """
    label, proba, explanation = predict_email_bert(email_text)
    
    return {
        "is_phishing": bool(label == 1),
        "probability": float(proba),
        "explanation": explanation  # List of (term, score) tuples
    }
