import json
from groq import Groq
from agents.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from agents.quality_agent.prompt import QUALITY_AGENT_SYSTEM, QUALITY_AGENT_USER


def run_quality_agent(context: dict) -> dict:
    """Execute the Quality Agent."""
    model_input = context["model_input"]
    model_output = context["model_output"]

    user_prompt = QUALITY_AGENT_USER.format(
        aco_id=context["aco_id"],
        year_t=context.get("year_t", "N/A"),
        current_quality_score=model_input.get("Current_Quality_Score", "N/A"),
        previous_quality_score=model_input.get("Previous_Quality_Score", "N/A"),
        readmission_rate=model_input.get("Readmission_Rate_Pct", "N/A"),
        patient_experience=model_input.get("Patient_Experience_Score", "N/A"),
        diabetes_control=model_input.get("Diabetes_Control_Rate_Pct", "N/A"),
        bp_control=model_input.get("Blood_Pressure_Control_Rate_Pct", "N/A"),
        preventive_screening=model_input.get("Preventive_Screening_Rate_Pct", "N/A"),
        followup_compliance=model_input.get("Followup_Compliance_Rate_Pct", "N/A"),
        predicted_quality_score=model_output.get("predicted_quality_score", "N/A"),
        quality_band=model_output.get("quality_band", "N/A"),
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": QUALITY_AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
