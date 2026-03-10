from __future__ import annotations

from app.schemas.execution import ExerciseRequest, ExerciseResponse
from app.services.exercise1_service import Exercise1Service
from app.services.exercise2_service import Exercise2Service
from app.services.exercise3_service import Exercise3Service
from app.services.exercise4_service import Exercise4Service
from app.services.exercise5_service import Exercise5Service
from app.services.exercise6_service import Exercise6Service


class ExerciseRunnerService:
    def __init__(self) -> None:
        self._services = {
            "ejercicio1": Exercise1Service(),
            "ejercicio2": Exercise2Service(),
            "ejercicio3": Exercise3Service(),
            "ejercicio4": Exercise4Service(),
            "ejercicio5": Exercise5Service(),
            "ejercicio6": Exercise6Service(),
        }

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        service = self._services.get(request.type.value)
        if service is None:
            raise ValueError(f"Tipo de ejercicio no soportado: {request.type.value}")
        return service.execute(request)
