import os
from datasets import load_dataset, concatenate_datasets

# 设置镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 清理缓存（可选）
# from huggingface_hub import HfApi
# api = HfApi()
# api.delete_cache()

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
