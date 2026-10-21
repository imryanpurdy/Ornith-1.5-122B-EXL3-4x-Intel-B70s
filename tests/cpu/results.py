"""Numbers recorded by tests, printed in the pytest terminal summary (see conftest.py)."""
RESULTS: list[tuple[str, object]] = []


def record(key: str, value) -> None:
    RESULTS.append((key, value))
