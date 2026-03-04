from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .protocols import AgentMission, LanguageModel, CandidateFetcher, ContactUpdater, RunStarter, RunFinisher
from .state import ScoutState
from .prompts import score_contact_prompt
from ._utils import parse_json_response


def create_scout_agent(
    llm: LanguageModel,
    fetch_candidates: CandidateFetcher,
    update_contact: ContactUpdater,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
    threshold: int = 60,
):
    """
    Build and return a compiled LangGraph scout agent.

    The agent fetches contacts with status='candidate', scores each for mission
    fit using the LLM, then promotes (status='cold') or drops (status='dropped')
    based on the score.

    Usage:
        agent = create_scout_agent(llm=..., fetch_candidates=..., ...)
        result = agent.invoke({"limit": 50})
        print(result["summary"])
    """

    def init(state: ScoutState) -> dict:
        run_id = start_run("scout_agent", {"limit": state.get("limit", 50)})
        return {
            "run_id": run_id,
            "limit": state.get("limit", 50),
            "candidates": [],
            "scores": [],
            "errors": [],
            "promoted_count": 0,
            "dropped_count": 0,
            "summary": "",
        }

    def fetch(state: ScoutState) -> dict:
        try:
            candidates = fetch_candidates(limit=state["limit"])
        except Exception as e:
            return {"errors": state["errors"] + [f"fetch_candidates: {e}"], "candidates": []}
        return {"candidates": candidates}

    def score_all(state: ScoutState) -> dict:
        if not state.get("candidates"):
            return {"scores": []}
        scores = []
        for contact in state["candidates"]:
            system, user = score_contact_prompt(mission, contact)
            try:
                response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                result = parse_json_response(response.content)
                scores.append({
                    "contact_id": contact["id"],
                    "score": int(result.get("score", 0)),
                    "reasoning": result.get("reasoning", ""),
                    "promote": bool(result.get("promote", False)),
                })
            except Exception as e:
                # score failure for one contact doesn't stop the batch
                scores.append({
                    "contact_id": contact["id"],
                    "score": 0,
                    "reasoning": f"scoring error: {e}",
                    "promote": False,
                })
        return {"scores": scores}

    def apply_scores(state: ScoutState) -> dict:
        promoted = 0
        dropped = 0
        for s in state.get("scores", []):
            try:
                new_status = "cold" if s["score"] >= threshold else "dropped"
                update_contact(
                    contact_id=s["contact_id"],
                    status=new_status,
                    fit_score=s["score"],
                    notes=s["reasoning"],
                )
                if s["promote"]:
                    promoted += 1
                else:
                    dropped += 1
            except Exception as e:
                pass
        return {"promoted_count": promoted, "dropped_count": dropped}

    def generate_report(state: ScoutState) -> dict:
        promoted = state.get("promoted_count", 0)
        dropped = state.get("dropped_count", 0)
        total = len(state.get("candidates", []))
        errs = state.get("errors", [])
        summary = (
            f"scout_agent: processed {total} candidates — "
            f"{promoted} promoted to cold, {dropped} dropped"
        )
        if errs:
            summary += f", {len(errs)} error(s)"
        finish_run(
            state.get("run_id", 0),
            "completed",
            summary,
            {"promoted": promoted, "dropped": dropped, "total": total},
        )
        return {"summary": summary}

    graph = StateGraph(ScoutState)
    graph.add_node("init", init)
    graph.add_node("fetch", fetch)
    graph.add_node("score_all", score_all)
    graph.add_node("apply_scores", apply_scores)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "fetch")
    graph.add_edge("fetch", "score_all")
    graph.add_edge("score_all", "apply_scores")
    graph.add_edge("apply_scores", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
