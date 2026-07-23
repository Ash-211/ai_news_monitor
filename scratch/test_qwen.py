import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

prompt = """You are an AI news verifier. Write a 2-sentence professional explanation for why an article received a specific credibility score based on the factors provided.

Article: Global Markets Fall Amid Tech Selloff
Score: 85%
Positive indicators: published by Reuters, a recognised and established news outlet.

Explanation:"""

messages = [{"role": "system", "content": "You are a professional AI news verification assistant."}, {"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.7, do_sample=True)
result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("Result:", result)
