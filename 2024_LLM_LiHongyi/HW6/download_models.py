# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v0_1")
model = AutoModelForCausalLM.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v0_1")