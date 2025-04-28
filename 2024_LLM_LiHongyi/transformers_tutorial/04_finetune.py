from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=5, torch_type='auto')
dataset = load_dataset("yelp_review_full")
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
metric = evaluate.load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()