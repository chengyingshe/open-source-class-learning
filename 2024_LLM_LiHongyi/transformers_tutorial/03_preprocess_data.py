from transformers import AutoTokenizer
from datasets import load_dataset, Audio

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]

encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)