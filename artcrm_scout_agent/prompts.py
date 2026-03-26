import json
from .protocols import AgentMission


def score_gallery_prompt(mission: AgentMission, contact: dict) -> tuple[str, str]:
    """
    Gallery-specific evaluation prompt. Reads the gallery's website content
    to determine whether they show emerging/regional/mid-career artists.
    """
    system = (
        f"You are researching art galleries on behalf of {mission.identity}.\n"
        f"Mission: {mission.goal}\n\n"
        f"Your task: read this gallery's website content carefully and determine "
        f"whether they are a realistic outreach target — meaning they actually show "
        f"emerging, regional, or mid-career artists.\n\n"
        f"What to look for:\n"
        f"- Names of artists they represent or have shown (are they internationally "
        f"established names, or regional/emerging artists?)\n"
        f"- How they describe themselves: 'zeitgenössisch', 'regional', 'Nachwuchs', "
        f"'emerging', 'junge Kunst', 'auf Kommission' are positive signals\n"
        f"- Exhibition language: rotating shows, open submissions, artist residencies "
        f"are positive signals\n"
        f"- Negative signals: exclusively internationally famous artists, blue-chip "
        f"focus, 'established masters only', auction house style\n"
        f"- If the website is thin or no content was fetched, use whatever is in "
        f"the notes field and mark as 'maybe'\n\n"
        f"Be specific in your reasoning — name artists you found, quote language "
        f"from the website, explain what tipped the decision."
    )
    contact_json = json.dumps(contact, ensure_ascii=False, indent=2)
    user = (
        f"Evaluate this gallery:\n\n"
        f"{contact_json}\n\n"
        f"Return a JSON object with EXACTLY these keys:\n"
        f"- outcome: one of 'cold' | 'maybe' | 'dropped'\n"
        f"  cold = shows emerging/regional/mid-career artists, realistic outreach target\n"
        f"  maybe = unclear, mixed signals, or website too thin to judge\n"
        f"  dropped = exclusively blue-chip/internationally established, not a fit\n"
        f"- reasoning: 3-5 sentences. Be specific — mention artist names, "
        f"exhibition titles, or quoted language from the website.\n\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user
