from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import inseq

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")
# List of attribution methods to be used
attribution_methods = ['saliency', 'attention']

for method in attribution_methods:
    # Load the model with the specified attribution method
    inseq_model = inseq.load_model(model, method)

    # Apply attribution to the input text using the specified method
    attribution_result = inseq_model.attribute(
        input_texts="The first president of America is",
        step_scores=["probability"],
    )

    # Remove 'Ġ' from GPT2's BPE tokenizer in the prefix to avoid confusion (You can ignore this part of code)
    for attr in attribution_result.sequence_attributions:
        for item in attr.source:
            item.token = item.token.replace('Ġ', '')
        for item in attr.target:
            item.token = item.token.replace('Ġ', '')

    # Display the attribution results
    attribution_result.show()