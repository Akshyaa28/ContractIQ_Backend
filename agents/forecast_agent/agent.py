import json
from groq import Groq
from agents.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from agents.forecast_agent.prompt import FORECAST_AGENT_SYSTEM, FORECAST_AGENT_USER


def run_forecast_agent(context: dict) -> dict:
    """Execute the Forecast Agent."""
    model_input = context["model_input"]
    model_output = context["model_output"]

    user_prompt = FORECAST_AGENT_USER.format(
        aco_id=context["aco_id"],
        n_ab=model_input["n_ab"],
        previous_savings_rate=model_input["previous_savings_rate"],
        previous_quality_score=model_input["previous_quality_score"],
        previous_performance_gap_pct=model_input["previous_performance_gap_pct"],
        expenditure_growth_pct=model_input["expenditure_growth_pct"],
        benchmark_growth_pct=model_input["benchmark_growth_pct"],
        beneficiary_growth_pct=model_input["beneficiary_growth_pct"],
        quality_change=model_input["quality_change"],
        forecasted_savings_rate_pct=model_output.get("forecasted_savings_rate_pct", "N/A"),
        savings_category=model_output.get("savings_category", "N/A"),
        savings_direction=model_output.get("savings_direction", "N/A"),
        model_r2_pct=model_output.get("model_r2_pct", 88.82),
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": FORECAST_AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    for key in ("key_findings", "improvement_areas", "recommended_actions"):
        if isinstance(result.get(key), list):
            flat = []
            for item in result[key]:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend([i for i in item if isinstance(i, str)])
            result[key] = flat

    return result
