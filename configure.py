# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Flag to determine which subject areas to use
flag = os.getenv("flag", "False") == "True"

if flag:
    subject_areas = os.getenv('subject_areas1', '').split(',')
else:
    subject_areas = os.getenv('subject_areas2', '').split(',')

# Select the first subject as the default
selected_subject = subject_areas[0] if subject_areas else None

# Load selected model and database configurations
models = os.getenv('models', '').split(',')
selected_models = models[0] if models else None

databases = os.getenv('databases', '').split(',')
selected_database = databases[0] if databases else None

# Gauge configuration for metrics
gauge_config = {
    "Faithfulness": {"value": 95, "color": "green"},
    "Relevancy": {"value": 82, "color": "lightgreen"},
    "Precision": {"value": 80, "color": "yellow"},
    "Recall": {"value": 78, "color": "orange"},
    "Harmfulness": {"value": 15, "color": "green"}
}
