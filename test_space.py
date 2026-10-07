import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print(f"Loaded HF_SPACE_FACTCHECK_URL: {os.environ.get('HF_SPACE_FACTCHECK_URL')}")

from src.intelligence.fake_news import call_local_fact_checker

test_prompt = """
Please evaluate the following news claim and output a JSON array exactly as requested.
Article Content: "A new study claims that drinking 10 liters of seawater a day cures all known diseases instantly."
Grammar Score: 0.2
RAG Evidence:
- WHO: Drinking seawater is extremely dangerous and causes severe dehydration and kidney failure.
- Medical Journal: No evidence supports seawater curing diseases; it is toxic in large amounts.
"""

print("\nSending prompt to custom Hugging Face Space API...")
result = call_local_fact_checker(test_prompt)

print("\n--- Model Response ---")
print(result)
