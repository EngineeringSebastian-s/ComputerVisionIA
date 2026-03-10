from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExerciseType(str, Enum):
    ejercicio1 = "ejercicio1"
    ejercicio2 = "ejercicio2"
    ejercicio3 = "ejercicio3"
    ejercicio4 = "ejercicio4"
    ejercicio5 = "ejercicio5"
    ejercicio6 = "ejercicio6"
    ejercicio6_plus = "ejercicio6_plus"


class ExerciseRequest(BaseModel):
    type: ExerciseType
    options: dict[str, Any] = Field(default_factory=dict)


class ImageArtifact(BaseModel):
    name: str
    path: str
    url: str
    content_base64: str


class ExerciseResponse(BaseModel):
    type: ExerciseType
    summary: dict[str, Any]
    images: list[ImageArtifact]
