from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypedDict

ControlType = Literal["slider", "int_slider", "select", "checkbox"]


@dataclass
class Option:
    """UI control descriptor for operation parameter"""

    name: str
    label: str
    control: ControlType

    min: float = None
    max: float = None
    step: float = None

    options: tuple[str, ...] | None = None
    help: str = None


ApplyFunction = Callable[[Any, dict[str, Any]], Any]


@dataclass
class Operation:
    """Describes image transformation with its function, metadata and parameters"""

    id: str
    label: str
    apply: ApplyFunction

    defaults: dict[str, Any] = field(default_factory=dict)
    params: tuple[Option, ...] = field(default_factory=tuple)

    experimental: bool = False
    description: str = None


class Step(TypedDict):
    """One operation step in pipeline"""

    id: str
    params: dict
