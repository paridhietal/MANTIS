import torch
from model import MantisModel, ProjectionHead
from augment import random_crop_resize
from loss import nt_xent_loss
from dataset import load_ucr_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MantisModel().to(device)
proj_head = ProjectionHead().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(proj_head.parameters()), lr=3e-4)

X_train, y_train, X_test, y_test = load_ucr_dataset("GunPoint")
batch_size = 32

for step in range(200):
    idx = torch.randint(0, X_train.size(0), (batch_size,))
    batch = X_train[idx].to(device)
    view1 = random_crop_resize(batch).to(device)
    view2 = random_crop_resize(batch).to(device)
    emb1, emb2 = model(view1), model(view2)
    z1, z2 = proj_head(emb1), proj_head(emb2)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# --- k-NN sanity check on the pretrained embeddings ---
model.eval()   # disable dropout for evaluation
with torch.no_grad():
    train_embeddings = model(X_train.to(device))   # (50, 256)
    test_embeddings = model(X_test.to(device))      # (150, 256)

def knn_predict(train_emb, train_labels, test_emb, k=5):
    # cosine similarity between every test embedding and every train embedding
    train_emb_norm = torch.nn.functional.normalize(train_emb, dim=-1)
    test_emb_norm = torch.nn.functional.normalize(test_emb, dim=-1)
    sim = test_emb_norm @ train_emb_norm.T          # (150, 50)

    topk_sim, topk_idx = sim.topk(k, dim=1)          # (150, k) -- k most similar train examples
    topk_labels = train_labels[topk_idx]              # (150, k) -- their labels

    preds = torch.mode(topk_labels, dim=1).values      # majority vote among k neighbors
    return preds

preds = knn_predict(train_embeddings.cpu(), y_train, test_embeddings.cpu(), k=5)
accuracy = (preds == y_test).float().mean().item()
print(f"\nk-NN accuracy on pretrained embeddings: {accuracy * 100:.2f}%")