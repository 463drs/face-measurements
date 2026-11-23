"""Known point layouts (schemas) and their sizes. Add an entry here when introducing a new layout."""

SCHEMA_NUM_POINTS: dict[str, int] = {
    "ibug68": 68,
    "yunet5": 5,
}


def num_points(schema: str) -> int:
    """Number of points in `schema`; raises KeyError for an unknown schema."""
    try:
        return SCHEMA_NUM_POINTS[schema]
    except KeyError:
        raise KeyError(f"unknown point schema {schema!r}; known: {sorted(SCHEMA_NUM_POINTS)}") from None
