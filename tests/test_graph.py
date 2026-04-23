from dataclasses import dataclass
from langchain_core.messages import AIMessage
from artcrm_scout_agent import create_scout_agent


@dataclass(frozen=True)
class DummyMission:
    goal: str = "Find art venues"
    identity: str = "Test Artist"
    targets: str = "galleries, cafes"
    fit_criteria: str = "contemporary art friendly"
    outreach_style: str = "personal"
    language_default: str = "de"


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index = 0

    def invoke(self, messages):
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return AIMessage(content=response)


GALLERY = {"id": 1, "name": "Galerie Nord", "city": "Munich", "type": "gallery", "status": "candidate"}
CAFE = {"id": 2, "name": "Café Mitte", "city": "Munich", "type": "cafe", "status": "candidate"}


def make_tools(candidates=None):
    updates = []
    runs = {}

    def fetch_candidates(limit=50):
        return [GALLERY] if candidates is None else candidates

    def update_contact(contact_id, status, fit_score, notes=""):
        updates.append({"id": contact_id, "status": status, "score": fit_score})

    def fetch_page(url):
        return "Contemporary art gallery showing regional artists."

    def fetch_city_context(city, country="DE"):
        return {"market_character": "regional", "market_notes": "Active local scene."}

    def start_run(agent_name, input_data):
        run_id = len(runs) + 1
        runs[run_id] = {"status": "running"}
        return run_id

    def finish_run(run_id, status, summary, output_data):
        runs[run_id]["status"] = status

    return fetch_candidates, update_contact, fetch_page, fetch_city_context, start_run, finish_run, updates, runs


def make_agent(candidates=None, llm_responses=None):
    fetch, update, fetch_page, fetch_city_context, start_run, finish_run, updates, runs = make_tools(candidates)
    llm = FakeLLM(llm_responses or ['{"outcome": "cold", "reasoning": "Good contemporary focus."}'])
    agent = create_scout_agent(
        llm=llm,
        fetch_candidates=fetch,
        update_contact=update,
        fetch_page=fetch_page,
        fetch_city_context=fetch_city_context,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )
    return agent, updates, runs


def test_gallery_cold_outcome():
    agent, updates, _ = make_agent(
        llm_responses=['{"outcome": "cold", "reasoning": "Shows emerging artists."}']
    )
    result = agent.invoke({"limit": 50})

    assert result["promoted_count"] == 1
    assert result["dropped_count"] == 0
    assert updates[0]["status"] == "cold"
    assert updates[0]["score"] == 75


def test_gallery_dropped_outcome():
    agent, updates, _ = make_agent(
        llm_responses=['{"outcome": "dropped", "reasoning": "Blue-chip only."}']
    )
    result = agent.invoke({"limit": 50})

    assert result["promoted_count"] == 0
    assert result["dropped_count"] == 1
    assert updates[0]["status"] == "dropped"
    assert updates[0]["score"] == 20


def test_gallery_maybe_outcome():
    agent, updates, _ = make_agent(
        llm_responses=['{"outcome": "maybe", "reasoning": "Unclear from website."}']
    )
    result = agent.invoke({"limit": 50})

    assert result["maybe_count"] == 1
    assert result["promoted_count"] == 0
    assert updates[0]["status"] == "maybe"
    assert updates[0]["score"] == 50


def test_non_gallery_auto_promoted():
    agent, updates, _ = make_agent(candidates=[CAFE])
    result = agent.invoke({"limit": 50})

    assert result["promoted_count"] == 1
    assert updates[0]["status"] == "cold"
    assert updates[0]["score"] == 50


def test_empty_candidates():
    agent, updates, _ = make_agent(candidates=[])
    result = agent.invoke({"limit": 50})

    assert result["promoted_count"] == 0
    assert result["dropped_count"] == 0
    assert updates == []


def test_score_parse_error_falls_back_to_maybe():
    candidates = [
        {"id": 1, "name": "Gallery A", "city": "Munich", "type": "gallery"},
        {"id": 2, "name": "Gallery B", "city": "Berlin", "type": "gallery"},
    ]
    agent, updates, _ = make_agent(
        candidates=candidates,
        llm_responses=["not json", '{"outcome": "cold", "reasoning": "Good fit."}'],
    )
    result = agent.invoke({"limit": 50})

    # first contact falls back to maybe, second promoted to cold
    assert result["maybe_count"] == 1
    assert result["promoted_count"] == 1
    assert updates[0]["status"] == "maybe"
    assert updates[1]["status"] == "cold"


def test_mixed_gallery_and_non_gallery():
    candidates = [GALLERY, CAFE]
    agent, updates, _ = make_agent(
        candidates=candidates,
        llm_responses=['{"outcome": "cold", "reasoning": "Shows emerging artists."}'],
    )
    result = agent.invoke({"limit": 50})

    # cafe auto-promoted + gallery scored cold
    assert result["promoted_count"] == 2
    statuses = {u["id"]: u["status"] for u in updates}
    assert statuses[1] == "cold"
    assert statuses[2] == "cold"
