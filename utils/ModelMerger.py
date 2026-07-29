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
    mergedName: str
    # Blend only the sample-rate-independent parts (encoder / flow / speaker
    # embedding) and take the decoder wholesale from the first model. This is the
    # only way to combine models trained at different sample rates.
    encoderOnly: bool = False


class ModelMerger(Protocol):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        ...
