# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("taide/TAIDE-LX-7B-Chat")