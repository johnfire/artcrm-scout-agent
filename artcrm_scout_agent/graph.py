"""
Scout agent.

Evaluates candidate contacts and classifies them as cold / maybe / dropped.
Non-gallery venues are auto-promoted to cold. Galleries are scored by LLM
using fetched website content and city market context.

Pipeline position: research → enrich → scout → outreach → followup
"""
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from .protocols import AgentMission, LanguageModel, CandidateFetcher, ContactUpdater, PageFetcher, CityContextFetcher, RunStarter, RunFinisher
from .prompts import score_gallery_prompt
from ._utils import parse_json_response

logger = logging.getLogger(__name__)

GALLERY_TYPES = {"gallery"}
_OUTCOME_SCORE = {"cold": 75, "maybe": 50, "dropped": 20}


class _ScoutAgent:
    def __init__(
        self,
        llm: LanguageModel,
        fetch_candidates: CandidateFetcher,
        update_contact: ContactUpdater,
        fetch_page: PageFetcher,
        fetch_city_context: CityContextFetcher,
        start_run: RunStarter,
        finish_run: RunFinisher,
        mission: AgentMission,
    ):
        self._llm = llm
        self._fetch_candidates = fetch_candidates
        self._update_contact = update_contact
        self._fetch_page = fetch_page
        self._fetch_city_context = fetch_city_context
        self._start_run = start_run
        self._finish_run = finish_run
        self._mission = mission

    def invoke(self, inputs: dict) -> dict:
        limit = inputs.get("limit", 50)
        run_id = self._start_run("scout_agent", {"limit": limit})

        candidates = self._fetch(limit)
        galleries, promoted_count = self._split_and_promote(candidates)
        galleries = self._fetch_gallery_websites(galleries)
        scores = self._score_galleries(galleries)
        promoted_count, maybe_count, dropped_count = self._apply_scores(scores, promoted_count)

        total = len(candidates)
        gallery_count = len(galleries)
        summary = (
            f"scout_agent: {total} candidates — "
            f"{promoted_count} promoted to cold, {maybe_count} flagged maybe, {dropped_count} dropped "
            f"({gallery_count} galleries evaluated by LLM)"
        )
        self._finish_run(
            run_id, "completed", summary,
            {"promoted": promoted_count, "maybe": maybe_count, "dropped": dropped_count, "total": total},
        )
        logger.info(summary)
        return {
            "summary": summary,
            "promoted_count": promoted_count,
            "maybe_count": maybe_count,
            "dropped_count": dropped_count,
        }

    def _fetch(self, limit: int) -> list[dict]:
        try:
            return self._fetch_candidates(limit=limit)
        except Exception as e:
            logger.warning("scout: fetch_candidates failed: %s", e)
            return []

    def _split_and_promote(self, candidates: list[dict]) -> tuple[list[dict], int]:
        """Auto-promote non-galleries to cold; return galleries for LLM evaluation."""
        promoted = 0
        galleries = []
        for contact in candidates:
            contact_type = (contact.get("type") or "").lower()
            if contact_type in GALLERY_TYPES:
                galleries.append(contact)
            else:
                try:
                    self._update_contact(
                        contact_id=contact["id"],
                        status="cold",
                        fit_score=50,
                        notes="Auto-promoted: non-gallery venue.",
                    )
                    promoted += 1
                except Exception:
                    pass
        return galleries, promoted

    def _fetch_gallery_websites(self, galleries: list[dict]) -> list[dict]:
        """Fetch website content for each gallery; cap at 4000 chars."""
        enriched = []
        for contact in galleries:
            contact = dict(contact)
            website = contact.get("website", "")
            website_content = ""
            if website:
                try:
                    website_content = self._fetch_page(website)[:4000]
                except Exception:
                    pass
            contact["website_content"] = website_content
            enriched.append(contact)
        return enriched

    def _score_galleries(self, galleries: list[dict]) -> list[dict]:
        """LLM evaluates each gallery; outcome is cold / maybe / dropped."""
        scores = []
        city_context_cache: dict[str, dict] = {}
        for contact in galleries:
            city = contact.get("city", "")
            country = contact.get("country", "DE")
            cache_key = f"{city}:{country}"
            if cache_key not in city_context_cache:
                try:
                    city_context_cache[cache_key] = self._fetch_city_context(city, country)
                except Exception:
                    city_context_cache[cache_key] = {}
            system, user = score_gallery_prompt(self._mission, contact, city_context_cache[cache_key])
            try:
                response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                result = parse_json_response(response.content)
                outcome = result.get("outcome", "maybe")
                if outcome not in ("cold", "maybe", "dropped"):
                    outcome = "maybe"
                scores.append({
                    "contact_id": contact["id"],
                    "outcome": outcome,
                    "reasoning": result.get("reasoning", ""),
                })
            except Exception as e:
                scores.append({
                    "contact_id": contact["id"],
                    "outcome": "maybe",
                    "reasoning": f"Scoring error — flagged for manual review: {e}",
                })
        return scores

    def _apply_scores(self, scores: list[dict], promoted_count: int) -> tuple[int, int, int]:
        """Write gallery outcomes to DB. Returns (promoted, maybe, dropped)."""
        maybe = 0
        dropped = 0
        for s in scores:
            try:
                self._update_contact(
                    contact_id=s["contact_id"],
                    status=s["outcome"],
                    fit_score=_OUTCOME_SCORE.get(s["outcome"], 50),
                    notes=s["reasoning"],
                )
                if s["outcome"] == "cold":
                    promoted_count += 1
                elif s["outcome"] == "maybe":
                    maybe += 1
                else:
                    dropped += 1
            except Exception:
                pass
        return promoted_count, maybe, dropped


def create_scout_agent(
    llm: LanguageModel,
    fetch_candidates: CandidateFetcher,
    update_contact: ContactUpdater,
    fetch_page: PageFetcher,
    fetch_city_context: CityContextFetcher,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
) -> _ScoutAgent:
    """
    Build and return a scout agent.

    Non-gallery contacts are auto-promoted to cold. Galleries are scored by
    LLM using fetched website content and city market context.

    Usage:
        agent = create_scout_agent(llm=..., fetch_candidates=..., ...)
        result = agent.invoke({"limit": 50})
        print(result["summary"])
    """
    return _ScoutAgent(
        llm=llm,
        fetch_candidates=fetch_candidates,
        update_contact=update_contact,
        fetch_page=fetch_page,
        fetch_city_context=fetch_city_context,
        start_run=start_run,
        finish_run=finish_run,
        mission=mission,
    )
