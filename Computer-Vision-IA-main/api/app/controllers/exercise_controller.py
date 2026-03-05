from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.execution import ExerciseRequest, ExerciseResponse
from app.services.exercise_runner_service import ExerciseRunnerService

router = APIRouter(prefix="/api/v1/exercises", tags=["Exercises"])
service = ExerciseRunnerService()


@router.post("/execute", response_model=ExerciseResponse)
def execute_exercise(payload: ExerciseRequest) -> ExerciseResponse:
    try:
        return service.execute(payload)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error ejecutando ejercicio: {exc}") from exc
