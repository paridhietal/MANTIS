import torch
from aeon.datasets import load_classification

def load_ucr_dataset(name="GunPoint"):
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")
    X_train = torch.tensor(X_train, dtype=torch.float32).squeeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).squeeze(1)
    X_train = (X_train - X_train.mean(dim=1, keepdim=True)) / (X_train.std(dim=1, keepdim=True) + 1e-8)
    X_test = (X_test - X_test.mean(dim=1, keepdim=True)) / (X_test.std(dim=1, keepdim=True) + 1e-8)
    classes = sorted(set(y_train))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = torch.tensor([class_to_idx[c] for c in y_train], dtype=torch.long)
    y_test = torch.tensor([class_to_idx[c] for c in y_test], dtype=torch.long)
    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_ucr_dataset("GunPoint")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape, "classes:", y_train.unique())
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("First series mean/std:", X_train[0].mean().item(), X_train[0].std().item())    