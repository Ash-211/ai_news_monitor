from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = """Article: Global Markets Fall Amid Tech Selloff
Source: Reuters
Credibility Score: 85%
Positive indicators: published by Reuters, a recognised and established news outlet, neutral objective tone.
Write a short, 2-sentence professional explanation for this credibility score."""
print("Result 1:", generate(prompt))

prompt2 = """Article: ALIENS LAND IN CENTRAL PARK!!!
Source: Unknown
Credibility Score: 15%
Risk factors: highly sensationalized title, excessive punctuation, high density of emotionally charged language.
Write a short, 2-sentence professional explanation for this credibility score."""
print("Result 2:", generate(prompt2))
