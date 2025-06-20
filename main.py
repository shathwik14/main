from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
import openai
from typing import Optional

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

model = joblib.load("telangana_xgboost_model.pkl")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreditRequest(BaseModel):
    age: int
    education_level: str
    employment_type: str
    job_stability: float
    family_dependents: int
    district: str
    monthly_income: int
    credit_score: Optional[int] = 0
    bank_statement_text: str

def get_repayment_percentage(income, job_stability, dependents, credit_score):
    income_score = min(income / 100000, 1.0)
    stability_score = min(job_stability / 20, 1.0)
    dependents_score = max(1 - (dependents / 6), 0)
    credit_score_norm = credit_score / 850 if credit_score > 0 else 0
    repayment_percent = (
        income_score * 0.4
        + stability_score * 0.2
        + dependents_score * 0.2
        + credit_score_norm * 0.2
    ) * 100
    return round(repayment_percent, 2)


def talk_with_llm(text: str) -> int:
    """
    Uses OpenAI GPT to analyze bank statement text.
    Returns a score adjustment from -10 to +10.
    """

    prompt = f"""
You are a finance expert. Analyze this bank statement:
\"\"\"{text}\"\"\"

Based on this, suggest a credit score adjustment:
Return a single number between -10 and +10 (no text, just the number).
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for loan analysis.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        output = response["choices"][0]["message"]["content"]
        return max(-10, min(10, int(output.strip())))

    except Exception as e:
        print("❌ GPT Error:", e)
        return 0


@app.post("/predict")
def predict_score(data: CreditRequest):
    input_data = data.dict()

    repayment = get_repayment_percentage(
        input_data["monthly_income"],
        input_data["job_stability"],
        input_data["family_dependents"],
        input_data["credit_score"],
    )

    input_data["repayment_capacity_percent"] = repayment
    bank_text = input_data.pop("bank_statement_text")

    input_df = pd.DataFrame([input_data])
    base_score = model.predict(input_df)[0]

    delta = talk_with_llm(bank_text)
    final_score = max(0, min(500, base_score + delta))

    return {
        "final_score": round(float(final_score), 2),
        "repayment_percent": round(float(repayment), 2),
    }
