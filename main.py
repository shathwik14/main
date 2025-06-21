from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client (new syntax)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the model
try:
    model = joblib.load("xgb_model_repayment_score_v2.pkl")
    EXPECTED_COLUMNS = list(model.feature_names_in_)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

app = FastAPI(title="Credit Scoring API", version="1.0.0")

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
    """Calculate repayment percentage based on income, stability, and dependents"""
    income_score = min(income / 100000, 1.0)
    stability_score = min(job_stability / 20, 1.0)
    dependents_score = max(1 - (dependents / 6), 0)
    repayment_percent = (
        income_score * 0.5 + stability_score * 0.3 + dependents_score * 0.2
    ) * 100
    return round(repayment_percent, 2)


def get_llm_adjustment(text: str) -> int:
    """Get credit score adjustment from LLM based on bank statement"""
    prompt = f"""
You are a finance expert. Analyze this bank statement:
\"\"\"{text}\"\"\"

Based on this, suggest a credit score adjustment:
Return a single number between -10 and +10 (no text, just the number).
"""
    try:
        response = client.chat.completions.create(
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
        output = response.choices[0].message.content
        return max(-10, min(10, int(output.strip())))
    except Exception as e:
        print(f"❌ GPT Error: {e}")
        return 0


def get_suggestions(input_data: dict, bank_text: str) -> str:
    """Get credit improvement suggestions from LLM"""
    suggestion_prompt = f"""
You are a financial advisor bot.

Here's a user's loan profile:
Employment Type: {input_data['employment_type']}
Education Level: {input_data['education_level']}
Job Stability (years): {input_data['job_stability_years']}
Family Dependents: {input_data['family_dependents']}
Monthly Income: ₹{input_data['monthly_income']}
District: {input_data['district']}
House Ownership: {input_data['house_ownership_status']}
Vehicle Ownership: {input_data['vehicle_ownership']}
Active Bank Accounts: {input_data['active_bank_accounts']}
Monthly Expenses: ₹{input_data['average_monthly_expenses']}
EMI Burden: ₹{input_data['existing_emi_burden']}
Savings Available: {input_data['emergency_savings_availability']}
Utility Bill Payments: {input_data['utility_bill_payment_regularity']}

Bank Statement Summary:
\"\"\"{bank_text}\"\"\"

👉 Based on this, give 2-3 suggestions to improve their credit score.
Keep it short and practical.

here are few suggestions:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart loan advisor who gives practical financial suggestions.",
                },
                {"role": "user", "content": suggestion_prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ GPT Error (suggestion): {e}")
        return "⚠️ Could not generate suggestions."


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Credit Scoring API is running!", "status": "healthy"}


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/predict")
def predict_score(data: CreditRequest):
    """Predict credit score based on user data"""

    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        input_data = data.dict()
        bank_text = input_data.pop("bank_statement_text")

        # Calculate repayment percentage
        repayment = get_repayment_percentage(
            input_data["monthly_income"],
            input_data["job_stability_years"],
            input_data["family_dependents"],
        )
        input_data["repayment_percent"] = repayment

        # Create DataFrame for model prediction
        input_df = pd.DataFrame([input_data])

        # Ensure all expected columns are present
        for col in EXPECTED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns

        # Reorder columns to match model expectation
        input_df = input_df[EXPECTED_COLUMNS]

        # Get base score from model
        base_score = model.predict(input_df)[0]

        # Get LLM adjustment
        delta = get_llm_adjustment(bank_text)

        # Calculate final score (ensure it's within bounds)
        final_score = max(0, min(500, base_score + delta))

        # Get suggestions
        suggestions = get_suggestions(input_data, bank_text)

        return {
            "final_score": round(float(final_score), 2),
            "repayment_percent": repayment,
            "suggestions": suggestions,
        }

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
