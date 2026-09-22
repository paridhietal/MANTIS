import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MantisModel
from stock_data import load_stock_windows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the pretrained encoder from checkpoint ---
checkpoint = torch.load("trained_mantis.pt", map_location=device)

model = MantisModel().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
for param in model.parameters():
    param.requires_grad = False   # freeze -- same reasoning as before, don't overfit

# --- Load real stock data ---
X_train, y_train, X_test, y_test = load_stock_windows(
    ticker="AAPL", lookback=60, horizon=5, target_len=512,
    start="2015-01-01", end="2024-01-01", split_date="2022-01-01"
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# --- Fresh classification head for THIS task (up/down), not reusing the old GunPoint head ---
num_classes = 2
classifier_head = nn.Linear(256, num_classes).to(device)
optimizer = torch.optim.Adam(classifier_head.parameters(), lr=1e-3)

batch_size = 64

for epoch in range(50):
    classifier_head.train()
    perm = torch.randperm(X_train.size(0))
    total_loss = 0.0
    
    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i + batch_size]
        batch_X, batch_y = X_train[idx], y_train[idx]
        
        with torch.no_grad():
            embeddings = model(batch_X)
            
        logits = classifier_head(embeddings)
        loss = F.cross_entropy(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if epoch % 5 == 0:
        classifier_head.eval()
        with torch.no_grad():
            test_embeddings = model(X_test)
            test_logits = classifier_head(test_embeddings)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
        
             # baseline: what if you always predicted the majority class?
            majority_class = y_train.float().mean().item() > 0.5
            baseline_preds = torch.full_like(y_test, int(majority_class))
            baseline_acc = (baseline_preds == y_test).float().mean().item()

        print(f"epoch {epoch:3d}  avg_loss {total_loss:.4f}  test_acc {test_acc*100:.2f}%  "
              f"(majority-class baseline: {baseline_acc*100:.2f}%)")