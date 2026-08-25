import torch
import torch.nn as nn 
import torch.nn.functional as F 

from tokenizer import MantisTokenizer
from encoder import ClsAndPositional, MantisEncoder

class MantisModel(nn.Module):
    def __init__(self, d_model=256, num_patches=32, num_heads=8,
                 num_layers=6, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.tokenizer = MantisTokenizer(d_model, num_patches)
        self.cls_pos = ClsAndPositional(d_model, max_len=num_patches + 1)
        self.encoder = MantisEncoder(d_model, num_heads, num_layers, ffn_dim, dropout)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = self.cls_pos(tokens)
        out = self.encoder(tokens)
        cls_embedding = out[:, 0, :]
        return cls_embedding
    
class ProjectionHead(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ) 
        
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)
    
if __name__ == "__main__":
    model = MantisModel()
    proj = ProjectionHead()
    
    dummy = torch.randn(8, 512)
    emb = model(dummy)
    z = proj(emb)
    print(emb.shape, z.shape)