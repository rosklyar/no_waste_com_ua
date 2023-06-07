from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Optimizer

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score as measure_f1_score

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