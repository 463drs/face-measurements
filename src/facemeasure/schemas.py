"""Known point layouts (schemas) and their sizes. Add an entry here when introducing a new layout."""

SCHEMA_NUM_POINTS: dict[str, int] = {
    "ibug68": 68,
    "yunet5": 5,
}


# ibug68 index layout (0-based) -> (contour index ranges, closed contour?).
# "Right"/"left" are the subject's: the right eye (36-41) is usually on the image's left.
IBUG68_REGIONS: dict[str, tuple[list[range], bool]] = {
    "jaw":   ([range(0, 17)], False),
    "brows": ([range(17, 22), range(22, 27)], False),           # right, left
    "nose":  ([range(27, 31), range(31, 36)], False),           # bridge, nostrils
    "eyes":  ([range(36, 42), range(42, 48)], True),            # right, left
    "mouth": ([range(48, 60), range(60, 68)], True),            # outer, inner lip
}


def num_points(schema: str) -> int:
    """Number of points in `schema`; raises KeyError for an unknown schema."""
    try:
        return SCHEMA_NUM_POINTS[schema]
    except KeyError:
        raise KeyError(f"unknown point schema {schema!r}; known: {sorted(SCHEMA_NUM_POINTS)}") from None
