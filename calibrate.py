import torch
import torch.nn as nn
import torch.nn.functional as F 

from model import MantisModel
from dataset import load_ucr_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the trained model + classifier head from disk
checkpoint = torch.load("trained_mantis.pt", map_location=device)

model = MantisModel().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
for param in model.parameters():
    param.requires_grad = False
    
classifier_head = nn.Linear(256, checkpoint["num_classes"]).to(device)
classifier_head.load_state_dict(checkpoint["classifier_head_state"])
classifier_head.eval()


# --- Load data ---
_, _, X_test, y_test = load_ucr_dataset("GunPoint")
X_test_dev, y_test_dev = X_test.to(device), y_test.to(device)

# 3. temperature scaling calibration

# split the test into validation/calibration half and a true held-out test half
n_test = X_test_dev.size(0)
perm = torch.randperm(n_test)
val_idx, holdout_idx = perm[:n_test // 2], perm[n_test // 2:]

X_val, y_val = X_test_dev[val_idx], y_test_dev[val_idx]
X_holdout, y_holdout = X_test_dev[holdout_idx], y_test_dev[holdout_idx]

#Get frozen logits on the validation split(no gradients needed for the encoder or head)
with torch.no_grad():
    val_embeddings = model(X_val)
    val_logits = classifier_head(val_embeddings)
    
#Temperature is the only thing we train here
temperature = torch.nn.Parameter(torch.ones(1, device=device))
temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

def temp_loss_closure():
    temp_optimizer.zero_grad()
    scaled_logits = val_logits / temperature
    loss = F.cross_entropy(scaled_logits, y_val)
    loss.backward()
    return loss

temp_optimizer.step(temp_loss_closure)
print(f"\nLearned temperature: {temperature.item():.4f}")

#Evaluation calibration on the held out half
with torch.no_grad():
    holdout_embeddings = model(X_holdout)
    holdout_logits = classifier_head(holdout_embeddings)
    
    uncalibrated_probs = F.softmax(holdout_logits, dim=1)
    calibrated_probs = F.softmax(holdout_logits / temperature, dim=1)
    
    preds = holdout_logits.argmax(dim=1)
    acc = (preds == y_holdout).float().mean().item()
    
    uncalibrated_confidence = uncalibrated_probs.max(dim=1).values.mean().item()
    calibrated_confidence = calibrated_probs.max(dim=1).values.mean().item()
    
    print(f"Held-out accuracy: {acc*100:.2f}%")
    print(f"Average confidence BEFORE calibration: {uncalibrated_confidence*100:.2f}%")
    print(f"Average confidence AFTER calibration: {calibrated_confidence*100:.2f}%")

