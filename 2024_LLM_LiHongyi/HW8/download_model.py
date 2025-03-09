# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

tokenizer = AutoTokenizer.from_pretrained("TheBloke/tulu-2-dpo-7B-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/tulu-2-dpo-7B-GPTQ")
print(f'*** Load model successfully!! ***')
