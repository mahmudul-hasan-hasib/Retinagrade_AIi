"""RetinaGrade AI - FastAPI application entrypoint.

Placeholder only. Dependencies are intentionally not installed yet.
Run with:
    .venv\Scripts\python.exe -m uvicorn main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="RetinaGrade AI API",
    version="0.1.0",
    description="Diabetic retinopathy screening API (CFP / UWF).",
)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "service": "retinagrade-ai"}
