from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Optimizer
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score as measure_f1_score
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.utils.saving import save_detector
from alibi_detect.cd import KSDrift
from functools import partial
from alibi_detect.cd.pytorch import HiddenOutput

class NoOpLogger:
    @staticmethod
    def log(*args, **kwargs):
        pass

def create_optimizer(model : nn.Module, optimizer_name: str, lr: float) -> Optimizer:
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(),
                        lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(),
                         lr=lr)
    return optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, device='cpu') -> float:
    cumu_loss = 0
    for _, batch in tqdm(enumerate(dataloader)):
        frames, labels = torch.squeeze(batch[0]).to(
            device), batch[1].to(device)
        optimizer.zero_grad()

        # forward pass
        loss = F.cross_entropy(model(frames).logits, labels)
        cumu_loss += loss.item()

        # backward pass
        loss.backward()
        optimizer.step()

    return cumu_loss / len(dataloader)

def save_drift_detector(model: nn.Module, dataloader: DataLoader, output_dir: Path, batch_size: int):
    drift_detector = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=batch_size)
    X_ref = torch.cat([data[0] for data in dataloader], dim=0)
    X_ref = torch.squeeze(X_ref).numpy()
    cd = KSDrift(x_ref=X_ref, preprocess_fn=drift_detector, p_val=.05, alternative='two-sided')
    save_detector(cd, output_dir / 'drift_detector')

@torch.no_grad()
def score_model(model: nn.Module, dataloader: DataLoader, device='cpu') -> float:
    print('Model scoring was started...')
    model.eval()
    dataloader.dataset.mode = 'eval'
    result = []
    targets = []
    for _, batch in tqdm(enumerate(dataloader)):
        frames = torch.squeeze(batch[0]).to(device)
        labels = batch[1].to(device)
        predicted = model(frames)
        predicted = predicted.logits.argmax(dim=-1)
        result.extend(predicted.cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    return measure_f1_score(targets, result, average='macro')