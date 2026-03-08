#Implement this as a google colab project or run locally without cells
# ── CELL 1: Mount Google Drive ──────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
# ── CELL 2: Install Dependencies ────────────────────────────
!pip install -q git+https://github.com/huggingface/diffusers transformers accelerate \
    bitsandbytes torchvision Pillow tqdm xformers peft

# Clone the official diffusers training scripts
!git clone https://github.com/huggingface/diffusers /content/diffusers_repo
# ── CELL 3: Configuration — EDIT THESE ──────────────────────
import os

DATASET_ROOT  = "/content/drive/MyDrive/Images"       # <-- CHANGE
OUTPUT_DIR    = "/content/processed_dataset/images"
TRAIN_OUT     = "/content/sd15_lora_finetuned"
DRIVE_SAVE    = "/content/drive/MyDrive/sd15_chicago_lora" # <-- YOU CAN CHANG THIS TOO IF YOU WANT

IMG_SIZE      = 512
BASE_MODEL    = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# DreamBooth uses a trigger word — the model learns to associate
# all your faces with this token
INSTANCE_PROMPT = "a photo of a chicagoface person"
CLASS_PROMPT    = "a photo of a person"          # generic class for prior preservation
CLASS_DIR       = "/content/class_images"        # auto-generated prior images

# Training
NUM_TRAIN_STEPS   = 1000    # for whole-dataset style learning, 800-1500 is good
BATCH_SIZE        = 1       # change this if you want, for me I couldn't have a bigger batch size due to VRAM constraints
GRADIENT_ACCUM    = 4       # effective batch = 4
LEARNING_RATE     = 1e-4    # higher LR works better for LoRA
MIXED_PRECISION   = "fp16"
SAVE_STEPS        = 250
NUM_CLASS_IMAGES  = 100     # prior preservation images to generate

# LoRA rank — higher = more expressive but more VRAM
# 16 is a good balance for a style/distribution task
LORA_RANK = 16
# ── CELL 4: Explore & Collect All Images ────────────────────
# I think you'll need to change this cell. I was using the Chicago Faces Dataset which is organzied in a very particular way, 
# and I wanted all the images in a single file for simplicity, so change this script depending on your needs.
import pathlib

root = pathlib.Path(DATASET_ROOT)
all_images = []  # (label, subject_id, image_path)

for split_dir in sorted(root.iterdir()):
    if not split_dir.is_dir():
        continue
    label = split_dir.name
    subject_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    print(f"  {label:20s}  →  {len(subject_dirs)} subjects")
    for subject_dir in subject_dirs:
        for img_file in subject_dir.iterdir():
            if img_file.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}:
                all_images.append((label, subject_dir.name, img_file))

print(f"\nTotal images found: {len(all_images)}")

# ── CELL 5: Center-Crop & Resize to 512×512 ─────────────────
from PIL import Image
from tqdm import tqdm

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def center_crop(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    img   = img.convert("RGB")
    w, h  = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img   = img.resize((new_w, new_h), Image.LANCZOS)
    left  = (new_w - size) // 2
    top   = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

os.makedirs(OUTPUT_DIR, exist_ok=True)
total_saved = 0

for (label, subject_id, img_path) in tqdm(all_images, desc="Processing"):
    try:
        img      = Image.open(img_path)
        img      = center_crop(img)
        out_name = f"{label}__{subject_id}.jpg"
        img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG", quality=95)
        total_saved += 1
    except Exception as e:
        print(f"  ⚠ Skipped {img_path.name}: {e}")

print(f"\n✅ Saved {total_saved} images to {OUTPUT_DIR}")

# ── CELL 6: Save Processed Dataset to Drive (do this on CPU runtime) ──
import shutil
DRIVE_PROCESSED = "/content/drive/MyDrive/chicago_faces_processed"
shutil.copytree(os.path.dirname(OUTPUT_DIR), DRIVE_PROCESSED, dirs_exist_ok=True)
print(f"✅ Saved to Drive → {DRIVE_PROCESSED}")

# ── CELL 7: Verify Sample Images ────────────────────────────
import matplotlib.pyplot as plt
import random
from PIL import Image

files  = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
sample = random.sample(files, min(9, len(files)))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, fname in zip(axes.flat, sample):
    img = Image.open(os.path.join(OUTPUT_DIR, fname))
    ax.imshow(img)
    ax.set_title(fname[:25], fontsize=7)
    ax.axis("off")
plt.suptitle("Sample Processed Images (512×512)")
plt.tight_layout()
plt.show()

# ── CELL 8: Generate Prior Preservation Class Images ────────
# These are generic "a photo of a person" images SD generates itself.
# They stop the model from forgetting what a general person looks like.

import torch
from diffusers import StableDiffusionPipeline

os.makedirs(CLASS_DIR, exist_ok=True)
existing = len([f for f in os.listdir(CLASS_DIR) if f.endswith(".jpg")])

if existing >= NUM_CLASS_IMAGES:
    print(f"✅ Already have {existing} class images, skipping generation")
else:
    needed = NUM_CLASS_IMAGES - existing
    print(f"Generating {needed} class images...")

    prior_pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    prior_pipe.set_progress_bar_config(disable=True)

    for i in tqdm(range(needed), desc="Class images"):
        image = prior_pipe(
            CLASS_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        image.save(os.path.join(CLASS_DIR, f"class_{existing + i:04d}.jpg"))

    del prior_pipe
    torch.cuda.empty_cache()
    print(f"✅ {NUM_CLASS_IMAGES} class images ready in {CLASS_DIR}")

# ── CELL 9: Write Accelerate Config ─────────────────────────
!accelerate config default --mixed_precision fp16

# ── CELL 10: Launch DreamBooth LoRA Training ────────────────
import os
os.makedirs(TRAIN_OUT, exist_ok=True)

cmd = f"""python /content/diffusers_repo/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="{BASE_MODEL}" \
  --instance_data_dir="{OUTPUT_DIR}" \
  --class_data_dir="{CLASS_DIR}" \
  --output_dir="{TRAIN_OUT}" \
  --instance_prompt="{INSTANCE_PROMPT}" \
  --class_prompt="{CLASS_PROMPT}" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --num_class_images={NUM_CLASS_IMAGES} \
  --resolution={IMG_SIZE} \
  --train_batch_size={BATCH_SIZE} \
  --gradient_accumulation_steps={GRADIENT_ACCUM} \
  --learning_rate={LEARNING_RATE} \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps={NUM_TRAIN_STEPS} \
  --checkpointing_steps={SAVE_STEPS} \
  --rank={LORA_RANK} \
  --mixed_precision="{MIXED_PRECISION}" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --seed=42"""

os.system(cmd)

# ── CELL 11: Save LoRA to Drive ──────────────────────────────
import shutil
shutil.copytree(TRAIN_OUT, DRIVE_SAVE, dirs_exist_ok=True)
print(f"✅ LoRA saved to Drive → {DRIVE_SAVE}")

# ── CELL 12: Generate 50 Images ─────────────────────────────
import torch, os
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.load_lora_weights(TRAIN_OUT)

os.makedirs("/content/generated", exist_ok=True)

prompt = "a photo of a chicagoface person, neutral expression, studio lighting"

images = []
for i in range(50):
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"/content/generated/face_{i:03d}.jpg")
    if i % 10 == 0:
        print(f"  Generated {i+1}/50...")

print("✅ Done — images saved to /content/generated/")

# Copy to Drive
import shutil
shutil.copytree("/content/generated", "/content/drive/MyDrive/chicago_generated", dirs_exist_ok=True)
print("✅ Copied to Drive")
