import sys
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, project_root)

from models.speech_pipeline.dataset import SpeechEmotionDataset
from models.speech_pipeline.model import SpeechBiLSTM

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = SpeechEmotionDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

loader = DataLoader(
    test_data,
    batch_size=32
)

model = SpeechBiLSTM().to(device)

model.load_state_dict(
    torch.load(
        "models/speech_pipeline/speech_model.pth",
        map_location=device
    )
)

model.eval()

y_true=[]
y_pred=[]

with torch.no_grad():

    for x,y in loader:

        x=x.to(device)

        outputs=model(x)

        preds=torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())


labels=[
"angry",
"disgust",
"fear",
"happy",
"neutral",
"surprise",
"sad"
]

cm=confusion_matrix(
    y_true,
    y_pred
)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(xticks_rotation=45)

plt.tight_layout()

plt.savefig(
    "Results/plots/speech_confusion_matrix.png"
)

plt.show()