"""
agents/sentiment_analyzer.py — Sentiment Analyzer Agent

Responsibilities:
- Detect the emotional tone of the customer's message
- Track how many consecutive turns the customer has been frustrated/angry
- These values feed directly into the escalation decision

This runs on EVERY turn so the escalation decider always has fresh data.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import SupportState
from backend.llm.client import get_llm_with_fallback


# ── Prompt ────────────────────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment analyzer for customer support conversations.

Analyze the customer's message and return a JSON object with:
1. "sentiment": one of these exact values:
   - "positive"   → happy, satisfied, thankful
   - "neutral"    → informational, calm, no strong emotion
   - "frustrated" → mildly upset, impatient, disappointed
   - "angry"      → very upset, aggressive, threatening to leave

2. "score": a float between -1.0 and 1.0 where:
   - 1.0  = very positive
   - 0.0  = neutral
   - -0.5 = frustrated
   - -1.0 = very angry

Return ONLY valid JSON. No explanation. Example:
{"sentiment": "frustrated", "score": -0.6}
"""


# ── Agent Node Function ────────────────────────────────────────────────────────

def sentiment_analyzer_node(state: SupportState) -> dict[str, Any]:
    """
    LangGraph node: analyzes sentiment of current user input.

    Reads:  state["current_input"], state["frustration_turns"], state["sentiment"]
    Writes: state["sentiment"], state["sentiment_score"], state["frustration_turns"]
    """
    import json

    user_input = state["current_input"]
    prev_frustration_turns = state.get("frustration_turns", 0)

    llm = get_llm_with_fallback(temperature=0.1)

    messages = [
        SystemMessage(content=SENTIMENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Customer message: {user_input}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        sentiment = parsed.get("sentiment", "neutral")
        score = float(parsed.get("score", 0.0))

    except Exception as e:
        print(f"[SentimentAnalyzer] Parsing failed: {e}")
        sentiment = "neutral"
        score = 0.0

    # Track consecutive frustrated/angry turns
    if sentiment in ("frustrated", "angry"):
        frustration_turns = prev_frustration_turns + 1
    else:
        frustration_turns = 0  # reset on positive/neutral

    print(f"[SentimentAnalyzer] Sentiment: {sentiment} (score: {score}) | Frustration turns: {frustration_turns}")

    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "frustration_turns": frustration_turns,
    }