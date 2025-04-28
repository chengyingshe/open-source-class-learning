from transformers import AutoTokenizer, AutoImageProcessor, AutoBackbone, AutoModelForCausalLM
from PIL import Image
import requests

# 1. Autotokenizer
# tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
# sequence = 'Hello, my name is Maxime!'
# output1 = tokenizer(sequence)
# print(output1)

# 2. AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# output2 = image_processor(image, return_tensors="pt")
# print(output2)

# 3. AutoBackbone
# model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
# output3 = model(**output2)
# print(output3)

# 4. AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('OpenGVLab/InternVL2-2B')