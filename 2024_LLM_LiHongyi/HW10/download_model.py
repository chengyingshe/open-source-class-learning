from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline
from transformers import CLIPTokenizer
import torch

pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae"
)
pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype= torch.bfloat16,
    safety_checker=None,
)