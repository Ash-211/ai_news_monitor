import sys
print("Starting script...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Loading model onto CPU with float32...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    print("Model loaded.")
    
    prompt = "Hello, write a 2 sentence story."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")
    
    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("Generated:", tokenizer.decode(outputs[0]))
    print("Done!")
except Exception as e:
    print(f"Caught Exception: {e}")
except BaseException as be:
    print(f"Caught BaseException: {be}")
sys.exit(0)
