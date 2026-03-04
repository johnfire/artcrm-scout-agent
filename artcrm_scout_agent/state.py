from typing import TypedDict


class ScoutState(TypedDict):
    # --- inputs ---
    limit: int          # max candidates to process per run

    # --- working state ---
    run_id: int
    candidates: list[dict]
    scores: list[dict]  # [{contact_id, score, reasoning, promote}]
    errors: list[str]

    # --- output ---
    promoted_count: int
    dropped_count: int
    summary: str
