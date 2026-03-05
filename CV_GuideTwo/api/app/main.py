from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.controllers.exercise_controller import router as exercise_router

app = FastAPI(
    title="Vision Computadora IA API",
    version="1.0.0",
    description="Capa IA escalable con FastAPI para ejercicios de Machine Learning.",
)

# CORS global para permitir consumo desde cualquier dominio/frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.mount("/static", StaticFiles(directory="app/output"), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(exercise_router)
