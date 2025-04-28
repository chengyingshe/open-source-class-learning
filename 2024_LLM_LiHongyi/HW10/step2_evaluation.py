import os
import cv2
import numpy as np
import torch
import glob
from deepface import DeepFace
from tqdm import tqdm


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
weight_dtype = torch.bfloat16 #
seed = 1126 # random seed
validation_prompt = "validation_prompt.txt"
output_folder = "logs" # 存放model checkpoints跟validation結果的資料夾
images_folder = "Datasets/Brad"
checkpoint_path = os.path.join(output_folder, f"checkpoint-last") # 設定使用哪個checkpoint inference
unet_path = os.path.join(checkpoint_path, "unet.pt")
text_encoder_path = os.path.join(checkpoint_path, "text_encoder.pt")
inference_path = "inference"
pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
os.makedirs(inference_path, exist_ok=True)
train_image_paths = []
for ext in IMAGE_EXTENSIONS:
    train_image_paths.extend(glob.glob(f"{images_folder}/*{ext}"))
train_image_paths = sorted(train_image_paths)
train_emb = torch.tensor([DeepFace.represent(img_path, detector_backend="ssd", model_name="GhostFaceNet", enforce_detection=False)[0]['embedding'] for img_path in train_image_paths])


def evaluate(pretrained_model_name_or_path, weight_dtype, seed, unet_path, text_encoder_path, validation_prompt, output_folder, train_emb):
    """
    (1) Goal:
        - This function is used to evaluate Stable Diffusion by loading UNet and Text Encoder from the given path and calculating face similarity, CLIP score, and the number of faceless images.

    (2) Arguments:
        - pretrained_model_name_or_path: str, model name from Hugging Face
        - weight_dtype: torch.type, model weight type
        - seed: int, random seed
        - unet_path: str, path to UNet model checkpoint
        - text_encoder_path: str, path to Text Encoder model checkpoint
        - validation_prompt: list, list of str storing texts for validation
        - output_folder: str, directory for saving generated images
        - train_emb: tensor, face features of training images

    (3) Returns:
        - output: face similarity, CLIP score, the number of faceless images

    """
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.unet = torch.load(unet_path)
    pipeline.text_encoder = torch.load(text_encoder_path)
    pipeline = pipeline.to(DEVICE)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = AutoModel.from_pretrained(clip_model_name)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)

    # run inference
    with torch.no_grad():
        generator = torch.Generator(device=DEVICE)
        generator = generator.manual_seed(seed)
        face_score = 0
        clip_score = 0
        mis = 0
        print("Generating validaion pictures ......")
        images = []
        for i in range(0, len(validation_prompt), 4):
            images.extend(pipeline(validation_prompt[i:min(i + 4, len(validation_prompt))], num_inference_steps=30, generator=generator).images)
        print("Calculating validaion score ......")
        valid_emb = []
        for i, image in enumerate(tqdm(images)):
            save_file = f"{output_folder}/valid_image_{i}.png"
            image.save(save_file)
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            emb = DeepFace.represent(
                opencvImage,
                detector_backend="ssd",
                model_name="GhostFaceNet",
                enforce_detection=False,
            )
            if emb == [] or emb[0]['face_confidence'] == 0:
                mis += 1
                continue
            emb = emb[0]
            inputs = clip_processor(text=validation_prompt[i], images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = clip_model(**inputs)
            sim = outputs.logits_per_image
            clip_score += sim.item()
            valid_emb.append(emb['embedding'])
        if len(valid_emb) == 0:
            return 0, 0, mis
        valid_emb = torch.tensor(valid_emb)
        valid_emb = (valid_emb / torch.norm(valid_emb, p=2, dim=-1)[:, None]).cuda()
        train_emb = (train_emb / torch.norm(train_emb, p=2, dim=-1)[:, None]).cuda()
        face_score = torch.cdist(valid_emb, train_emb, p=2).mean().item()
        # face_score = torch.min(face_score, 1)[0].mean()
        clip_score /= len(validation_prompt) - mis
    return face_score, clip_score, mis


face_score, clip_score, mis = evaluate(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    weight_dtype=weight_dtype,
    seed=seed,
    unet_path=unet_path,
    text_encoder_path=text_encoder_path,
    validation_prompt=validation_prompt,
    output_folder=inference_path,
    train_emb=train_emb,
)
print("Face Similarity Score:", face_score, "CLIP Score:", clip_score, "Faceless Images:", mis)