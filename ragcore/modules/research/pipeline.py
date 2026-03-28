"""Research pipeline - Orchestrates complete research workflow."""

import logging
import uuid
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import Job, Session as SessionModel
from ragcore.modules.research.models import (
    ResearchSessionState,
    ResearchFinding,
    ToolCall,
    ToolResult,
    ResearchTurn,
)
from ragcore.modules.research.tool_registry import executor
from ragcore.modules.research.agent_planner import ResearchPlanner

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Orchestrates multi-turn research workflow with tool execution."""

    def __init__(self):
        """Initialize research pipeline."""
        self.planner = ResearchPlanner()
        self.max_turns = settings.research_max_turns
        self.max_results = settings.research_max_results_per_search

    async def research(
        self,
        query: str,
        session_id: UUID,
        research_mode: str = "standard",  # "standard" | "deep"
        file_ids: Optional[List[UUID]] = None,
    ) -> tuple[str, List[dict], ResearchSessionState]:
        """
        Execute complete research workflow.

        Multi-turn loop:
        1. Plan next action (agent reasoning)
        2. Execute search tool(s)
        3. Synthesize findings
        4. Check: continue or finalize?

        Args:
            query: Research query
            session_id: Session identifier
            research_mode: "standard" (web search) or "deep" (GPT Researcher)
            file_ids: Optional file IDs for compound mode

        Returns:
            Tuple of (final_response, sources_list, session_state)
        """
        logger.info(
            f"Starting research pipeline: query='{query[:50]}' "
            f"mode={research_mode} session={session_id}"
        )

        # Initialize session state
        session_state = ResearchSessionState(
            session_id=session_id,
            turns=[],
            findings_summary={},
            agent_decisions=[],
        )

        try:
            # =====================================================================
            # Multi-Turn Research Loop
            # =====================================================================

            while session_state.should_continue(self.max_turns):
                turn_num = session_state.current_turn + 1
                logger.info(f"Research turn {turn_num}/{self.max_turns}")

                # -----------------------------------------------------------------
                # Step 1: Plan Next Action (Agent Reasoning)
                # -----------------------------------------------------------------

                findings_summary = self._summarize_findings(session_state)
                decision = await self.planner.plan_next_action(
                    session_state=session_state,
                    current_query=query,
                    current_findings_summary=findings_summary,
                )

                session_state.record_decision(
                    decision=decision["decision"],
                    reasoning=decision.get("reasoning", ""),
                    metadata={"mode": research_mode},
                )

                # -----------------------------------------------------------------
                # Step 2: Check Agent Decision
                # -----------------------------------------------------------------

                if not self.planner.should_continue_research(session_state, decision["decision"]):
                    logger.info(
                        f"Agent decided to finalize. Decision: {decision['decision']}"
                    )
                    session_state.research_complete = True
                    break

                # -----------------------------------------------------------------
                # Step 3: Execute Search (with Tool Selection)
                # -----------------------------------------------------------------

                # Determine next search query
                next_query = decision.get("next_query") or query
                if decision["decision"] == "search_different":
                    # Agent suggested different angle
                    next_query = f"{query} - {decision.get('reasoning', '')[:50]}"

                logger.info(f"Executing search: '{next_query[:60]}'")

                # Create tool call record
                tool_call = ToolCall(
                    id=str(uuid.uuid4())[:8],
                    tool_name="web_search",
                    query=next_query,
                    status="executing",
                )

                try:
                    # Select tool and execute with fallback
                    results, tool_used = await executor.execute_with_fallback(
                        query=next_query,
                        max_results=self.max_results,
                    )

                    tool_call.update_status(
                        status="completed",
                        result=[r.to_dict() for r in results],
                    )

                    logger.info(
                        f"Search completed: tool={tool_used}, results={len(results)}"
                    )

                except Exception as e:
                    logger.error(f"Search execution error: {str(e)}")
                    tool_call.update_status(
                        status="failed",
                        error=str(e),
                    )
                    results = []
                    tool_used = "none"

                # -----------------------------------------------------------------
                # Step 4: Create Research Finding
                # -----------------------------------------------------------------

                # Synthesize this search into findings
                finding_synthesis = self._synthesize_search_results(
                    query=next_query,
                    results=[r.to_dict() for r in results],
                )

                finding = ResearchFinding(
                    query=next_query,
                    results=[r.to_dict() for r in results],
                    tool_used=tool_used,
                    synthesis=finding_synthesis,
                    executed_at=datetime.utcnow(),
                    confidence_score=self._calculate_confidence(len(results)),
                )

                # -----------------------------------------------------------------
                # Step 5: Create Research Turn
                # -----------------------------------------------------------------

                turn = ResearchTurn(
                    role="assistant",
                    content=finding_synthesis,
                    created_at=datetime.utcnow(),
                    sources=file_ids,
                    tool_calls=[tool_call],
                    tool_results=[],
                    research_findings=[finding],
                    agent_reasoning=decision.get("reasoning"),
                )

                session_state.add_turn(turn)

                logger.debug(f"Turn {turn_num} completed: {turn}")

            # =====================================================================
            # Final Synthesis & Response
            # =====================================================================

            logger.info(f"Research complete after {session_state.current_turn} turns")

            # Collect all findings
            all_findings = list(session_state.findings_summary.values())

            # Synthesize final response
            final_response = self.planner.synthesize_findings(
                findings_list=all_findings,
                query=query,
            )

            # Extract sources for response
            sources = self._extract_sources(all_findings)

            # Update session in database
            await self._update_session_with_research(
                session_id=session_id,
                research_state=session_state,
            )

            logger.info(f"Research pipeline complete: {len(all_findings)} findings, {len(sources)} sources")

            return final_response, sources, session_state

        except Exception as e:
            logger.error(f"Research pipeline error: {str(e)}", exc_info=True)

            # Return partial results if available
            all_findings = list(session_state.findings_summary.values())
            final_response = (
                f"Research encountered an error: {str(e)}. "
                f"Partial findings: {len(all_findings)} searches completed."
            )
            sources = self._extract_sources(all_findings)

            return final_response, sources, session_state

    def _summarize_findings(self, session_state: ResearchSessionState) -> str:
        """Summarize findings so far for agent planning."""
        if not session_state.findings_summary:
            return "No findings yet."

        summary_lines = []
        for finding in list(session_state.findings_summary.values()):
            summary_lines.append(
                f"• Query: {finding.query}\n"
                f"  Tool: {finding.tool_used}\n"
                f"  Results: {len(finding.results)}\n"
                f"  Synthesis: {finding.synthesis[:100]}..."
            )

        return "\n".join(summary_lines)

    def _synthesize_search_results(self, query: str, results: List[dict]) -> str:
        """Synthesize search results into brief finding."""
        if not results:
            return f"No results found for '{query}'."

        # Extract snippets
        snippets = [r.get("snippet", "")[:100] for r in results[:3]]
        synthesis = f"Found {len(results)} results for '{query}': " + " | ".join(
            [s for s in snippets if s]
        )

        return synthesis[:300]

    def _calculate_confidence(self, result_count: int) -> float:
        """Calculate confidence score based on result count."""
        if result_count >= 5:
            return 0.9
        elif result_count >= 3:
            return 0.75
        elif result_count > 0:
            return 0.6
        else:
            return 0.2

    def _extract_sources(self, findings: List[ResearchFinding]) -> List[dict]:
        """Extract unique sources from findings."""
        sources_set = {}

        for finding in findings:
            for result in finding.results:
                url = result.get("url", "")
                if url and url not in sources_set:
                    sources_set[url] = {
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": result.get("snippet", "")[:100],
                        "source": finding.tool_used,
                    }

        return list(sources_set.values())

    async def _update_session_with_research(
        self,
        session_id: UUID,
        research_state: ResearchSessionState,
    ) -> None:
        """Update session with research findings."""
        from ragcore.db.database import async_session_factory

        async with async_session_factory() as session:
            try:
                stmt = select(SessionModel).where(SessionModel.id == session_id)
                result = await session.execute(stmt)
                session_record = result.scalar_one_or_none()

                if session_record:
                    # Store research state in session (JSON field if available)
                    # For now, this is a placeholder for future schema extension
                    logger.debug(f"Updated session {session_id} with research findings")
                    await session.commit()

            except Exception as e:
                logger.warning(f"Failed to update session: {str(e)}")


# Global pipeline instance
pipeline = ResearchPipeline()
