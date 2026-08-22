import json
from groq import Groq
from agents.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from agents.risk_agent.prompt import RISK_AGENT_SYSTEM, RISK_AGENT_USER


def run_risk_agent(context: dict) -> dict:
    """
    Execute the Risk Agent.
    context must have: aco_id, model_input, model_output
    Returns structured JSON analysis.
    """
    model_input = context["model_input"]
    model_output = context["model_output"]

    user_prompt = RISK_AGENT_USER.format(
        aco_id=context["aco_id"],
        n_ab=model_input["n_ab"],
        previous_savings_rate=model_input["previous_savings_rate"],
        previous_quality_score=model_input["previous_quality_score"],
        previous_performance_gap_pct=model_input["previous_performance_gap_pct"],
        expenditure_growth_pct=model_input["expenditure_growth_pct"],
        benchmark_growth_pct=model_input["benchmark_growth_pct"],
        beneficiary_growth_pct=model_input["beneficiary_growth_pct"],
        quality_change=model_input["quality_change"],
        risk_probability_pct=model_output.get("risk_probability_pct", "N/A"),
        risk_label=model_output.get("risk_label", "N/A"),
        risk_level=model_output.get("risk_level", "N/A"),
        threshold=model_output.get("threshold", 0.23),
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": RISK_AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Fix malformed output: LLM sometimes dumps everything into key_findings
    # Detect pattern: if key_findings contains strings like "attention_areas", ":", etc.
    if isinstance(result.get("key_findings"), list) and len(result.get("key_findings", [])) > 5:
        # LLM likely merged all fields into key_findings — re-parse
        items = result["key_findings"]
        clean_findings = []
        attention = []
        actions = []
        confidence = ""
        current_bucket = "findings"

        for item in items:
            if not isinstance(item, str):
                continue
            item_lower = item.strip().lower().rstrip(":")
            if item_lower in ("attention_areas", "attention areas"):
                current_bucket = "attention"
                continue
            elif item_lower in ("recommended_actions", "recommended actions"):
                current_bucket = "actions"
                continue
            elif item_lower in ("confidence_note", "confidence note"):
                current_bucket = "confidence"
                continue
            elif item_lower == ":":
                continue

            if current_bucket == "findings":
                clean_findings.append(item)
            elif current_bucket == "attention":
                attention.append(item)
            elif current_bucket == "actions":
                actions.append(item)
            elif current_bucket == "confidence":
                confidence = item if not confidence else confidence + " " + item

        result["key_findings"] = clean_findings if clean_findings else result.get("key_findings", [])[:3]
        if attention:
            result["attention_areas"] = attention
        if actions:
            result["recommended_actions"] = actions
        if confidence:
            result["confidence_note"] = confidence

    # Ensure expected keys exist with correct types
    expected_lists = ["key_findings", "attention_areas", "recommended_actions"]
    for key in expected_lists:
        if not isinstance(result.get(key), list) or not result[key]:
            result[key] = result.get(key, []) or ["No specific items identified."]
        # Flatten nested lists
        flat = []
        for item in result[key]:
            if isinstance(item, str) and item.strip():
                flat.append(item)
            elif isinstance(item, list):
                flat.extend([i for i in item if isinstance(i, str) and i.strip()])
        result[key] = flat if flat else ["No specific items identified."]

    if not result.get("confidence_note"):
        result["confidence_note"] = "Model prediction based on historical patterns. Ongoing monitoring advised."

    return result
