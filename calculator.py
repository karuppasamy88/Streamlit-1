"""Core calculator logic used by the Streamlit UI."""


def calculate_percentage_change(original: float, updated: float) -> float:
    """Return percentage change from original to updated.

    Raises:
        ValueError: If ``original`` is zero (percentage change undefined).
    """
    if original == 0:
        raise ValueError("Original value cannot be zero.")
    return ((updated - original) / original) * 100
