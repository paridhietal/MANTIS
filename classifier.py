import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model import MantisModel, ProjectionHead
from augment import random_crop_resize
from loss import nt_xent_loss
from dataset import load_ucr_dataset
from stock_pool import build_financial_pretraining_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MantisModel().to(device)
proj_head = ProjectionHead().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(proj_head.parameters()), lr=3e-4)

TOTAL_STEPS = 1500
WARMUP_STEPS = 100

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Building financial pretraining pool...")
pretrain_pool = build_financial_pretraining_pool()

X_train, y_train, X_test, y_test = load_ucr_dataset("GunPoint")   # kept for reference, unused in this run
batch_size = 32

for step in range(TOTAL_STEPS):
    idx = torch.randint(0, pretrain_pool.size(0), (batch_size,))
    batch = pretrain_pool[idx].to(device)
    view1 = random_crop_resize(batch).to(device)
    view2 = random_crop_resize(batch).to(device)
    emb1, emb2 = model(view1), model(view2)
    z1, z2 = proj_head(emb1), proj_head(emb2)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"[pretrain] step {step:4d}  loss {loss.item():.4f}  lr {current_lr:.6f}")
        
# --- Phase 2: freeze encoder, fine-tune on real stock up/down labels ---
for param in model.parameters():
    param.requires_grad = False
model.eval()

from stock_data import load_stock_windows
X_train, y_train, X_test, y_test = load_stock_windows(
    ticker="AAPL", lookback=60, horizon=5, target_len=512,
    start="2015-01-01", end="2024-01-01", split_date="2022-01-01"
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

num_classes = 2
classifier_head = nn.Linear(256, num_classes).to(device)
clf_optimizer = torch.optim.Adam(classifier_head.parameters(), lr=1e-3)

stock_batch_size = 64

for epoch in range(50):
    classifier_head.train()
    perm = torch.randperm(X_train.size(0))
    total_loss = 0.0

    for i in range(0, X_train.size(0), stock_batch_size):
        idx = perm[i:i + stock_batch_size]
        batch_X, batch_y = X_train[idx], y_train[idx]
        with torch.no_grad():
            embeddings = model(batch_X)
        logits = classifier_head(embeddings)
        loss = F.cross_entropy(logits, batch_y)
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        classifier_head.eval()
        with torch.no_grad():
            test_logits = classifier_head(model(X_test))
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            majority_class = y_train.float().mean().item() > 0.5
            baseline_preds = torch.full_like(y_test, int(majority_class))
            baseline_acc = (baseline_preds == y_test).float().mean().item()
        print(f"[finetune] epoch {epoch:3d}  avg_loss {total_loss:.4f}  test_acc {test_acc*100:.2f}%  (baseline: {baseline_acc*100:.2f}%)")

torch.save({
    "model_state": model.state_dict(),
    "classifier_head_state": classifier_head.state_dict(),
    "num_classes": num_classes,
}, "trained_mantis_finance.pt")
print("\nSaved to trained_mantis_finance.pt")        