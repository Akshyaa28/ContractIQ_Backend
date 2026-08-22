import json
from groq import Groq
from agents.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from agents.recommendation_agent.prompt import RECOMMENDATION_SYSTEM, RECOMMENDATION_USER


def run_recommendation_agent(aco_id: str, agent_outputs: dict) -> dict:
    """
    Execute the Recommendation Agent.
    Synthesizes outputs from available specialized agents.

    agent_outputs: dict of available analyses, e.g.:
        {"risk": {...}, "forecast": {...}, "twin": {...}}
    """
    available = json.dumps(agent_outputs, indent=2)

    user_prompt = RECOMMENDATION_USER.format(
        aco_id=aco_id,
        available_analyses=available,
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
