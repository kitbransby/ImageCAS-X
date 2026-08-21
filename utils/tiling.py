def axis_starts(dim: int, patch: int, step: int) -> list:
    """Tile positions along one axis, last window snapped to the edge."""
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, step))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts
