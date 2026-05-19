import sys
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset

dataset = SpeechEmotionDataset()

loader = DataLoader(
    dataset,
    batch_size=100
)

features=[]
labels=[]

for x,y in loader:
    features.extend(
        x.view(x.size(0),-1).numpy()
    )

    labels.extend(
        y.numpy()
    )

features = np.array(features[:1000])
labels = np.array(labels[:1000])

tsne=TSNE(
    n_components=2,
    random_state=42
)

reduced=tsne.fit_transform(features)

plt.figure(figsize=(10,8))

scatter=plt.scatter(
    reduced[:,0],
    reduced[:,1],
    c=labels
)

plt.colorbar(scatter)

plt.savefig(
    "Results/plots/tsne_speech.png"
)

plt.show()