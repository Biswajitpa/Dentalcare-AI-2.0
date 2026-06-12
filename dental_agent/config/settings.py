import os
from pathlib import Path

# Setup core workspace directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project root directory
CSV_PATH = str(BASE_DIR / "doctor_availability.csv")

# Free Local Model Configurations
MODEL_NAME = "llama3.1"
TEMPERATURE = 0.0

VALID_SPECIALIZATIONS = [
    "general_dentist",
    "oral_surgeon",
    "orthodontist",
    "cosmetic_dentist",
    "prosthodontist",
    "pediatric_dentist",
    "emergency_dentist"
]

VALID_DOCTORS = [
    "john doe",
    "emily johnson",
    "sarah wilson",
    "jane smith",
    "michael green",
    "robert martinez",
    "lisa brown",
    "susan davis",
    "daniel miller",
    "kevin anderson"
]

DATE_FORMAT = "%m/%d/%Y %H:%M"