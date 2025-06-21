from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = joblib.load("xgb_model_repayment_score.pkl")
EXPECTED_COLUMNS = list(model.feature_names_in_)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreditRequest(BaseModel):
    employment_type: str
    education_level: str
    job_stability_years: float
    family_dependents: int
    monthly_income: int
    district: str
    house_ownership_status: str
    vehicle_ownership: str
    active_bank_accounts: int
    average_monthly_expenses: int
    existing_emi_burden: int
    emergency_savings_availability: str
    utility_bill_payment_regularity: str
    bank_statement_text: str

def get_repayment_percentage(income, job_stability, dependents):
    income_score = min(income / 100000, 1.0)
    stability_score = min(job_stability / 20, 1.0)
    dependents_score = max(1 - (dependents / 6), 0)
    repayment_percent = (
        income_score * 0.5 + stability_score * 0.3 + dependents_score * 0.2
    ) * 100
    return round(repayment_percent, 2)

def talk_with_llm(text: str) -> int:
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
                {"role": "system", "content": "You are a helpful assistant for loan analysis."},
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
    bank_text = input_data.pop("bank_statement_text")

    repayment = get_repayment_percentage(
        input_data["monthly_income"],
        input_data["job_stability_years"],
        input_data["family_dependents"],
    )
    input_data["repayment_percent"] = repayment

    input_df = pd.DataFrame([input_data])
    base_score = model.predict(input_df)[0]

    delta = talk_with_llm(bank_text)
    final_score = max(0, min(500, base_score + delta))

    return {
        "final_score": round(float(final_score), 2),
        "repayment_percent": repayment,
        "llm_adjustment": delta,
    }
