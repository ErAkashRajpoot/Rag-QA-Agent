import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings

def create_app() -> FastAPI:
    """Bootstraps FastAPI backend dependencies automatically"""
    app = FastAPI(title="Agentic RAG Microservice API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def global_health():
        return {"status": "up", "environment": settings.ENVIRONMENT, "hf_model": settings.MODEL_NAME}

    return app

api_handler = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:api_handler", host=settings.API_ADDRESS, port=settings.API_PORT, reload=(settings.ENVIRONMENT=="development"))
