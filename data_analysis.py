!pip install deepface pandas tqdm matplotlib
from google.colab import drive
drive.mount('/content/drive')
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from deepface import DeepFace
BASE_DIR = "/content/drive/MyDrive/faces_experiment"

groups = {
    "real": os.path.join(BASE_DIR, "real"),
    "finetuned": os.path.join(BASE_DIR, "finetuned_model"),
    "pretrained": os.path.join(BASE_DIR, "pretrained_model")
}
for group, path in groups.items():
    print(group, ":", len(os.listdir(path)), "images")
results = []

for group_name, folder in groups.items():
    
    images = os.listdir(folder)
    
    for img_name in tqdm(images, desc=group_name):
        
        img_path = os.path.join(folder, img_name)

        try:
            analysis = DeepFace.analyze(
                img_path,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion_scores = analysis[0]["emotion"]

            # proxy attractiveness score
            score = (
                emotion_scores["happy"] +
                emotion_scores["neutral"]
            )

            results.append({
                "image": img_name,
                "group": group_name,
                "attractiveness_score": score
            })

        except Exception as e:
            print("Skipped:", img_name)
df = pd.DataFrame(results)

print(df.head())
save_path = "/content/drive/MyDrive/attractiveness_results.csv"

df.to_csv(save_path, index=False)

print("Saved results to:", save_path)
group_means = df.groupby("group")["attractiveness_score"].mean()

print(group_means)
group_means.plot(kind="bar")

plt.ylabel("Average Attractiveness Score")
plt.title("Attractiveness Comparison Across Face Sources")

plt.show()
