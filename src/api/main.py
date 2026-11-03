import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

app = FastAPI(
    title="Scalable Fraud Detection & Explainable AI Engine",
    description="Production-grade Data Engineering pipeline, LightGBM fraud model & SHAP XAI REST API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)

# Mount Dashboard static directory
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard')
if os.path.exists(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
