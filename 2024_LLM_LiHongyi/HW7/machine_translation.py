# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import inseq

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# List of attribution methods to be used
attribution_methods = ['saliency', 'attention']

for method in attribution_methods:
    print(f"======= Attribution Method: {method} =======")
    # Load the Chinese-to-English translation model and set up the attribution method
    insqe_model = inseq.load_model("Helsinki-NLP/opus-mt-zh-en", method)

    # Apply attribution to the input text using the specified method
    attribution_result = insqe_model.attribute(
        input_texts="我喜歡機器學習和人工智慧。",
        step_scores=["probability"],
    )

    # Remove '▁' from the tokenizer in the prefix to avoid confusion (You can ignore this part of code)
    for attr in attribution_result.sequence_attributions:
        for item in attr.source:
            item.token = item.token.replace('▁', '')
        for item in attr.target:
            item.token = item.token.replace('▁', '')

    # Display the attribution results
    attribution_result.show()
    