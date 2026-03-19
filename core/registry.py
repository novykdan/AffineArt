from .schemes import Operation


_REGISTRY: dict[str, Operation] = {}


def register_operation(operation: Operation) -> None:
    """Register operation object under its id in global registry"""
    if operation.id in _REGISTRY:
        raise ValueError(f"Operation with id '{operation.id}' is already registered")
    _REGISTRY[operation.id] = operation


def get_operation(operation_id: str) -> Operation:
    """Return registered operation for operation_id or raise KeyError"""
    try:
        return _REGISTRY[operation_id]
    except KeyError as err:
        registered = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown operation id: {operation_id}. Registered: [{registered}]"
        ) from err


def list_operations() -> list[Operation]:
    """List all registred operation objects"""
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """Remove all operations from global registry"""
    _REGISTRY.clear()
