import json
from .protocols import AgentMission


def score_contact_prompt(mission: AgentMission, contact: dict) -> tuple[str, str]:
    system = (
        f"You are evaluating potential clients for {mission.identity}.\n"
        f"Mission: {mission.goal}\n"
        f"Fit criteria: {mission.fit_criteria}"
    )
    user = (
        f"Evaluate this contact as a potential client for outreach:\n"
        f"{json.dumps(contact, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON object with:\n"
        f"- score: integer 0-100 (how well they match the fit criteria)\n"
        f"- reasoning: one sentence explaining the score\n\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user
