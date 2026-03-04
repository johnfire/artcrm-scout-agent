# artcrm-scout-agent

LangGraph agent that scores candidate contacts for mission fit. Promotes good matches to `status=cold` and drops poor ones.

## What it does

Fetches all contacts with `status=candidate`, scores each with the LLM against the mission's `fit_criteria`, then:
- Score >= 60: promotes to `status=cold` (ready for outreach)
- Score < 60: sets to `status=dropped` with reasoning logged

Individual scoring failures don't stop the batch — the run continues.

## Usage

```python
from artcrm_scout_agent import create_scout_agent

agent = create_scout_agent(
    llm=your_llm,
    fetch_candidates=your_fetch_fn,
    update_contact=your_update_fn,
    start_run=your_start_run_fn,
    finish_run=your_finish_run_fn,
    mission=your_mission,
)

result = agent.invoke({"limit": 50})
print(result["summary"])
# "scout_agent: processed 15 candidates — 9 promoted to cold, 6 dropped"
```

## Protocols

All dependencies are injected. Each callable must match the Protocol defined in [protocols.py](artcrm_scout_agent/protocols.py):

| Parameter | Protocol | Description |
|---|---|---|
| `llm` | `LanguageModel` | Any LangChain `BaseChatModel` |
| `fetch_candidates` | `CandidateFetcher` | `(limit: int) -> list[dict]` |
| `update_contact` | `ContactUpdater` | `(contact_id, status, fit_score, notes) -> None` |
| `start_run` | `RunStarter` | `(agent_name, input_data) -> int` |
| `finish_run` | `RunFinisher` | `(run_id, status, summary, output_data) -> None` |
| `mission` | `AgentMission` | Any object with the six mission fields |

## Testing

```bash
uv run pytest -v
```

## Support

If you find this useful, a small donation helps keep projects like this going:
[Donate via PayPal](https://paypal.me/christopherrehm001)
