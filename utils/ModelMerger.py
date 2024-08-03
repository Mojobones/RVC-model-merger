from typing import Protocol
from dataclasses import dataclass


@dataclass
class MergeElement:
    modelPath: str
    strength: int


@dataclass
class ModelMergerRequest:
    command: str
    files: list[MergeElement]


class ModelMerger(Protocol):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        ...
