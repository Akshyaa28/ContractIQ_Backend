import json
from groq import Groq
from agents.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from agents.twin_agent.prompt import TWIN_AGENT_SYSTEM, TWIN_AGENT_USER


def run_twin_agent(context: dict) -> dict:
    """Execute the Twin Agent."""
    model_input = context["model_input"]
    model_output = context["model_output"]

    # Format twins for the prompt
    twins = model_output.get("twins", [])
    twins_str = json.dumps(twins, indent=2) if twins else "No twins found."

    user_prompt = TWIN_AGENT_USER.format(
        aco_id=context["aco_id"],
        n_ab=model_input["n_ab"],
        previous_savings_rate=model_input["previous_savings_rate"],
        previous_quality_score=model_input["previous_quality_score"],
        previous_performance_gap_pct=model_input["previous_performance_gap_pct"],
        expenditure_growth_pct=model_input["expenditure_growth_pct"],
        benchmark_growth_pct=model_input["benchmark_growth_pct"],
        beneficiary_growth_pct=model_input["beneficiary_growth_pct"],
        quality_change=model_input["quality_change"],
        twins_json=twins_str,
        top5_avg=model_output.get("top5_avg_savings_rate_pct", "N/A"),
        top5_median=model_output.get("top5_median_savings_rate_pct", "N/A"),
        top5_best=model_output.get("top5_best_savings_rate_pct", "N/A"),
        top5_worst=model_output.get("top5_worst_savings_rate_pct", "N/A"),
        outperformer_count=model_output.get("outperformer_count", 0),
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": TWIN_AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    for key in ("key_similarities", "outperforming_twins", "benchmark_insights", "recommended_actions"):
        if isinstance(result.get(key), list):
            flat = []
            for item in result[key]:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend([i for i in item if isinstance(i, str)])
            result[key] = flat

    return result
