from fastapi import FastAPI
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import json, os
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from contextlib import asynccontextmanager

# Load Supabase credentials
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE_NAME = "eligible_mutualfunds_clients"
HISTORY_TABLE = "recommendation_history"

class Person(BaseModel):
    Person_Name: str = Field(..., alias="Person's Name")
    Account_Number: int = Field(..., alias="Account Number")
    Phone_Number: int = Field(..., alias="Phone Number")
    Age: int
    Date_of_Birth: str = Field(..., alias="Date of Birth")
    Education: str
    Loans: str
    Mutual_Funds: str = Field(..., alias="Mutual Funds or Investments")
    Employment_Type: str = Field(..., alias="Employment Type")
    Income: int
    Bank_Branch_Name: str = Field(..., alias="Bank Branch Name")
    Marital_Status: str = Field(..., alias="Marital Status")
    Email_Address: str = Field(..., alias="Email Address")
    Account_Type: str = Field(..., alias="Account Type")
    Online_Banking: str = Field(..., alias="Online Banking Enabled")
    SMS_Banking: str = Field(..., alias="SMS Banking Enabled")

    @field_validator('Account_Number')
    def validate_account_number(cls, v):
        if str(v).startswith('0') or not (6 <= len(str(v)) <= 20):
            raise ValueError("Account Number must be between 6 and 20 digits and not start with 0")
        return v

    @field_validator('Phone_Number')
    def validate_phone_number(cls, v):
        if len(str(v)) != 10:
            raise ValueError("Phone Number must be exactly 10 digits")
        return v

    @field_validator('Date_of_Birth')
    def validate_dob_format(cls, v):
        try:
            datetime.strptime(v, "%d/%m/%Y")
            return v
        except ValueError:
            raise ValueError("Date of Birth must be in DD/MM/YYYY format")

    class Config:
        validate_by_name = True   # replaces allow_population_by_field_name


# ML Model Globals
clf = None
label_encoders = {}

async def train_model_and_clear_supabase():
    global clf, label_encoders

    print("Loading and training the model...")

    with open("labeled_data.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # drop personal identifiers
    df.drop(columns=["Account Number", "Phone Number", "Person's Name", "Date of Birth", "Bank Branch Name", "Email Address"], inplace=True)

    # encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        if col != "Eligible":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    df["Eligible"] = df["Eligible"].map({"Yes": 1, "No": 0})

    X = df.drop("Eligible", axis=1)
    y = df["Eligible"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test))
    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n" + report)

    # clear Supabase eligible clients table
    print("Clearing Supabase eligible client table...")
    supabase.table(TABLE_NAME).delete().neq("id", 0).execute()


# FastAPI lifespan (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    await train_model_and_clear_supabase()  # runs at startup
    yield
    # (You can add shutdown cleanup here if needed)


app = FastAPI(lifespan=lifespan)


@app.post("/check_eligibility/")
def check_eligibility(people: List[Person]):
    global clf, label_encoders
    results = []

    for person in people:
        try:
            features = pd.DataFrame([{
                'Age': person.Age,
                'Education': person.Education,
                'Loans': person.Loans,
                'Mutual Funds or Investments': person.Mutual_Funds,
                'Employment Type': person.Employment_Type,
                'Income': person.Income,
                'Marital Status': person.Marital_Status,
                'Account Type': person.Account_Type,
                'Online Banking Enabled': person.Online_Banking,
                'SMS Banking Enabled': person.SMS_Banking
            }])

            # apply encoders
            for col, le in label_encoders.items():
                features[col] = le.transform(features[col])

            prediction = clf.predict(features)[0]

            dob_iso = datetime.strptime(person.Date_of_Birth, "%d/%m/%Y").strftime("%Y-%m-%d")
            payload = person.dict(by_alias=True)
            payload["Eligible"] = "Yes" if prediction == 1 else "No"
            payload["Date of Birth"] = dob_iso

            if prediction == 1:
                # Insert into eligible clients table
                supabase.table(TABLE_NAME).insert(payload).execute()

                # Remove old record in recommendation_history (if exists)
                supabase.table(HISTORY_TABLE).delete().eq("Account Number", person.Account_Number).execute()

                # Insert into recommendation history table
                supabase.table(HISTORY_TABLE).insert(payload).execute()

                results.append({
                    "Person": person.Person_Name,
                    "status": "Eligible",
                    "message": "Person is eligible and added to Supabase"
                })

            else:
                reasons = []
                if person.Mutual_Funds != "No Investments":
                    reasons.append("Person already has investments")
                if person.Income <= 150000:
                    reasons.append("Income is less than or equal to ₹150,000")

                results.append({
                    "Person": person.Person_Name,
                    "status": "Not Eligible",
                    "message": "Person is not eligible",
                    "reasons": reasons
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "Person": person.Person_Name,
                "status": "Error",
                "message": "Processing failed",
                "error": str(e)
            })

    return results
