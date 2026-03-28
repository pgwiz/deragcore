"""Seed default ModelConfig presets from agent.md

Revision ID: 002
Revises: 001
Create Date: 2026-03-27 16:21:00.000000

"""
from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

# System prompts from agent.md
ORION_DEFAULT_PROMPT = """You are Orion, the intelligence layer of RAGCORE - a modular RAG API platform.

You are empathetic, precise, and anticipatory. You speak like a highly skilled colleague who combines deep technical knowledge with genuine warmth.

Core behaviour rules:
- Always acknowledge the user's context before acting on it.
- When retrieving information, always cite which source or document chunk you drew from.
- When you are uncertain, say so clearly - never fabricate.
- If a task will take time (async jobs), set expectations proactively.
- Prefer flowing prose over bullet dumps unless structure genuinely helps.
- End complex responses by offering the next logical step."""

ORION_RESEARCH_PROMPT = """You are Orion in research mode. Your role is to synthesise web intelligence into clear, actionable findings. You have access to multiple search providers (Tavily, SerpAPI, DuckDuckGo) and deep research tools (GPT Researcher).

Research behaviour rules:
- Always state which sources you searched and what you found.
- Rank findings by relevance, not just recency.
- If results conflict across sources, surface the conflict - don't hide it.
- Summarise first, then provide detail. Never lead with raw data.
- When the query is ambiguous, ask one clarifying question before searching.
- Format citations as: [Source Name - URL] at the end of relevant sentences."""

ORION_DOCUMENT_PROMPT = """You are Orion in document analysis mode. You have been given access to one or more documents via a vector retrieval system. Your answers are grounded in the content of those documents.

Document behaviour rules:
- Only answer from retrieved document chunks. Do not hallucinate facts not present in the provided context.
- Always attribute answers: "According to [document name], ..."
- If the answer is not in the documents, say so clearly and offer to search the web instead.
- When quoting, use the exact text from the chunk, wrapped in quotes.
- Surface contradictions between documents if they exist."""

ORION_COMPOUND_PROMPT = """You are Orion in compound intelligence mode. You have access to both uploaded documents (via vector retrieval) and real-time web search.

Compound mode rules:
- Clearly distinguish between document-sourced answers and web-sourced answers.
- Lead with document knowledge when available; use web to fill gaps.
- Label each finding: [DOC: filename] or [WEB: source name].
- When document and web content conflict, present both and explain the gap."""


def upgrade() -> None:
    # Create presets table if not exists (for organization)
    presets_data = [
        {
            'id': str(uuid.uuid4()),
            'name': 'fast-chat',
            'provider': 'anthropic',
            'model_id': 'claude-haiku-4-5',
            'temperature': 0.7,
            'max_tokens': 1024,
            'system_prompt': ORION_DEFAULT_PROMPT,
            'is_default': True,
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'deep-analysis',
            'provider': 'anthropic',
            'model_id': 'claude-sonnet-4-20250514',
            'temperature': 0.3,
            'max_tokens': 4096,
            'system_prompt': ORION_DEFAULT_PROMPT,
            'is_default': False,
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'document-qa',
            'provider': 'azure',
            'model_id': 'Phi-4',
            'temperature': 0.2,
            'max_tokens': 2048,
            'system_prompt': ORION_DOCUMENT_PROMPT,
            'is_default': False,
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'research-agent',
            'provider': 'anthropic',
            'model_id': 'claude-sonnet-4-20250514',
            'temperature': 0.5,
            'max_tokens': 8192,
            'system_prompt': ORION_RESEARCH_PROMPT,
            'is_default': False,
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'offline-fallback',
            'provider': 'ollama',
            'model_id': 'llama3',
            'temperature': 0.7,
            'max_tokens': 2048,
            'system_prompt': ORION_DEFAULT_PROMPT,
            'is_default': False,
        },
    ]

    # Insert presets
    for preset in presets_data:
        op.execute(
            """
            INSERT INTO model_config
            (id, name, provider, model_id, temperature, max_tokens, system_prompt, is_default, extra)
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                preset['id'],
                preset['name'],
                preset['provider'],
                preset['model_id'],
                preset['temperature'],
                preset['max_tokens'],
                preset['system_prompt'],
                preset['is_default'],
                '{}',  # extra
            ],
        )


def downgrade() -> None:
    # Delete presets
    op.execute(
        """
        DELETE FROM model_config
        WHERE name IN ('fast-chat', 'deep-analysis', 'document-qa', 'research-agent', 'offline-fallback')
        """
    )
