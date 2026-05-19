import os
import pandas as pd

data = []

dataset_path = "data/TESS"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        emotion = "_".join(folder.split("_")[1:]).lower()

        if emotion in ["pleasant_surprised", "pleasant_surprise"]:
            emotion = "surprise"

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                text = file.replace(".wav", "")
                parts = text.split("_")

                    # Remove speaker name and emotion word
                text = " ".join(parts[1:-1])

                data.append([text, emotion])

df = pd.DataFrame(data, columns=["text", "emotion"])

df.to_csv("data/text_dataset.csv", index=False)

print(df.head())
print("\nTotal samples:", len(df))
print("\nEmotion counts:")
print(df["emotion"].value_counts())