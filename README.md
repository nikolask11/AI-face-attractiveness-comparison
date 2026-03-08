# AI Face Attractiveness Experiment  
## Dataset Bias vs Feature Averaging in Generative Models

This project investigates **why AI-generated faces often appear more attractive than real human faces**.

Two competing explanations are tested:

1. **Dataset Bias**  
   AI models trained on large internet datasets may learn idealized facial features.

2. **Feature Averaging**  
   Generative models may statistically average facial structures, producing aesthetically pleasing faces.

To test this, we fine-tune a diffusion model on a **small unbiased human face dataset** and compare the attractiveness of:

- Real human faces
- Fine-tuned model outputs
- Pretrained model outputs

---

# Project Overview

The pipeline performs the following steps:

1. Load and preprocess the Chicago Face Database
2. Fine-tune a diffusion model using **DreamBooth + LoRA**
3. Generate synthetic faces
4. Score attractiveness using a computer vision model
5. Compare attractiveness between groups

The training pipeline is designed to run on **Google Colab free-tier GPUs**.

---

# Technologies Used

- **Diffusers** — diffusion model training framework  
- **Stable Diffusion 1.5** — base generative model  
- **DreamBooth** — fine-tuning technique  
- **LoRA (Low-Rank Adaptation)** — efficient parameter training  
- **DeepFace** — face analysis library  
- **Chicago Face Database** — real human face dataset  

---

# Hardware Requirements

Minimum recommended environment:

**Platform:** Google Colab Free Tier  

**GPU**
NVIDIA T4 (16GB VRAM)


**RAM**

12GB


**Disk space**

~10GB

#Implementation
Open a new colab notebook and paste each of the cells in and run. I recommend running cells 1-7 on a CPU, and save the results on drive, and then running the rest on a GPU runtime. After generating the images, run the data analysis script with images. 
