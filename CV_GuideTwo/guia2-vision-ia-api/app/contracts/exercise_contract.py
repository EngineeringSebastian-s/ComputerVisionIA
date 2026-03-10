from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.execution import ExerciseRequest, ExerciseResponse


class ExerciseContract(ABC):
    @property
    @abstractmethod
    def exercise_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        raise NotImplementedError
