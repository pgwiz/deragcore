"""Agent planner - Multi-turn research reasoning and decision making."""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from ragcore.config import settings
from ragcore.core.ai_controller import AIController
from ragcore.modules.research.models import (
    ResearchSessionState,
    ResearchFinding,
    ResearchTurn,
)

logger = logging.getLogger(__name__)

# System prompt for research planning
RESEARCH_PLANNER_PROMPT = """You are Orion's research planner. Your job is to analyze research progress and decide the next action.

Given the current research session, you must decide:
- "search_more": The query needs clarification or additional searches to fully answer
- "search_different": Try a different search strategy or angle
- "finalize": We have enough information to synthesize a final response

Respond with JSON: {"decision": "...", "reasoning": "...", "next_query": "..." or null}"""


class ResearchPlanner:
    """Plans multi-turn research workflows with agent reasoning."""

    def __init__(self):
        """Initialize planner."""
        self.provider = "anthropic"  # Use Claude for planning
        self.model = "claude-3-5-sonnet-20241022"
        self.max_turns = settings.research_max_turns
        self.temperature = 0.7

    async def plan_next_action(
        self,
        session_state: ResearchSessionState,
        current_query: str,
        current_findings_summary: str,
    ) -> Dict[str, Any]:
        """
        Decide next action in research workflow using agentic reasoning.

        Args:
            session_state: Current research session state
            current_query: Original user query
            current_findings_summary: Summary of findings so far

        Returns:
            Decision dict with: {decision, reasoning, next_query}
        """
        logger.debug(f"Planning next action for research session {session_state.session_id}")

        # Build planning context
        planning_context = f"""
Research Progress:
- Original Query: {current_query}
- Research Turns: {session_state.current_turn}/{self.max_turns}
- Tool Calls: {session_state.total_tool_calls}
- Findings: {len(session_state.findings_summary)} unique findings

Current Findings Summary:
{current_findings_summary}

Recent Agent Decisions:
{json.dumps(session_state.agent_decisions[-3:] if session_state.agent_decisions else [], indent=2)}
"""

        planning_message = f"""Based on the research progress below, what should we do next?

{planning_context}

Decide: search_more | search_different | finalize
Explain your reasoning.
If searching more, provide the next search query.
"""

        try:
            # Use AIController for planning decision
            response = AIController.complete(
                provider_name=self.provider,
                model_id=self.model,
                messages=[{"role": "user", "content": planning_message}],
                temperature=self.temperature,
                max_tokens=500,
                system_prompt=RESEARCH_PLANNER_PROMPT,
            )

            # Parse response - attempt to extract JSON decision
            response_text = response.text

            # Try to extract JSON from response
            try:
                # Find JSON block in response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    decision_json = json.loads(json_match.group())
                else:
                    # Fallback: parse as structured text
                    decision_json = self._parse_text_decision(response_text)

            except json.JSONDecodeError:
                logger.warning("Failed to parse decision as JSON, parsing as text")
                decision_json = self._parse_text_decision(response_text)

            logger.info(
                f"Agent decided: {decision_json['decision']} - {decision_json['reasoning'][:80]}"
            )

            return decision_json

        except Exception as e:
            logger.error(f"Planning error: {str(e)}")
            # Graceful fallback: finalize if error
            return {
                "decision": "finalize",
                "reasoning": f"Planning error: {str(e)}",
                "next_query": None,
            }

    def _parse_text_decision(self, text: str) -> Dict[str, Any]:
        """Parse decision from plain text response."""
        text_lower = text.lower()

        if "finalize" in text_lower:
            decision = "finalize"
        elif "search_different" in text_lower or "different" in text_lower:
            decision = "search_different"
        elif "search_more" in text_lower or "more" in text_lower:
            decision = "search_more"
        else:
            decision = "finalize"  # Default to finalize

        return {
            "decision": decision,
            "reasoning": text[:200],
            "next_query": None,
        }

    def should_continue_research(
        self,
        session_state: ResearchSessionState,
        decision: str,
    ) -> bool:
        """
        Check if research workflow should continue.

        Args:
            session_state: Current session state
            decision: Agent decision from plan_next_action

        Returns:
            True if should continue, False if should finalize
        """
        # Check max turns
        if session_state.current_turn >= self.max_turns:
            logger.info(f"Reached max research turns ({self.max_turns})")
            return False

        # Check agent decision
        if decision == "finalize":
            logger.info("Agent decided to finalize research")
            return False

        # Continue for "search_more" or "search_different"
        return True

    def synthesize_findings(
        self,
        findings_list: list[ResearchFinding],
        query: str,
    ) -> str:
        """
        Synthesize multiple research findings into coherent narrative.

        Args:
            findings_list: List of ResearchFinding objects
            query: Original research query

        Returns:
            Synthesized narrative string
        """
        if not findings_list:
            return "No findings from research."

        # Build findings context
        findings_context = "\n\n".join(
            [
                f"Search Query: {f.query}\nTool: {f.tool_used}\nResults: {len(f.results)}\n"
                f"Synthesis: {f.synthesis}"
                for f in findings_list
            ]
        )

        synthesis_message = f"""Based on these research findings, provide a coherent, well-structured synthesis.

Original Query: {query}

Findings:
{findings_context}

Requirements:
1. Organize findings by theme/topic
2. Cite sources: [Source Name - URL]
3. Surface any conflicts or contradictions
4. Provide clear takeaways
5. Keep to 2-3 paragraphs maximum
"""

        try:
            response = AIController.complete(
                provider_name=self.provider,
                model_id=self.model,
                messages=[{"role": "user", "content": synthesis_message}],
                temperature=0.7,
                max_tokens=1000,
                system_prompt="You are Orion, synthesizing research findings into clear narratives.",
            )

            synthesized = response.text
            logger.info(f"Synthesized {len(findings_list)} findings into narrative")
            return synthesized

        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            # Fallback: list findings
            return "\n\n".join(
                [f"• {f.query} ({f.tool_used}): {f.synthesis[:100]}..." for f in findings_list]
            )

    def get_research_summary_for_context(
        self,
        session_state: ResearchSessionState,
    ) -> str:
        """
        Generate research summary for inclusion in chat context.

        Args:
            session_state: Current session state

        Returns:
            Summary string for embedding in message context
        """
        summary_lines = [
            "Research Summary:",
            f"  Turns: {session_state.current_turn}",
            f"  Tool Calls: {session_state.total_tool_calls}",
            f"  Unique Findings: {len(session_state.findings_summary)}",
        ]

        if session_state.research_findings:
            summary_lines.append("\n  Key Findings:")
            for finding in list(session_state.findings_summary.values())[:3]:
                snippet = finding.synthesis[:80] + ("..." if len(finding.synthesis) > 80 else "")
                summary_lines.append(f"    • {snippet}")

        return "\n".join(summary_lines)
