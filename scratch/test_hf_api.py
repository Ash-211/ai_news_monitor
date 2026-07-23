from huggingface_hub import InferenceClient

client = InferenceClient()

prompt = "Article: Global Markets Fall Amid Tech Selloff\nScore: 85%\nPositive indicators: established news outlet.\nWrite a short, professional 2-sentence explanation for this credibility score."

try:
    response = client.chat_completion(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
    )
    print("API Result:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
