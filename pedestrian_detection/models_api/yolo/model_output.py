from dataclasses import dataclass


@dataclass
class ModelOutput:
    x: int
    y: int
    w: int
    h: int
    conf: float
    class_id: int
