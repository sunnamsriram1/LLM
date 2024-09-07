import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set environment variable for memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Consider using a smaller model
try:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    model, tokenizer = None, None

# Function to generate a response from GPT-Neo
def get_gpt_neo_response(prompt, max_new_tokens=50):
    if model and tokenizer:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,  # Temperature works when do_sample=True
                do_sample=True,   # Enable sampling to use temperature
                pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    else:
        return "Model not loaded."

# Interactive loop
def interactive_loop():
    print("Welcome to the GPT-Neo interactive prompt! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_gpt_neo_response(prompt, max_new_tokens=50)
        print(f"GPT-Neo: {response}")

# Run the interactive loop
interactive_loop()
