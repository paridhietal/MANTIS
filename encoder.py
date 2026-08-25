import math
import torch
import torch.nn as nn

class ClsAndPositional(nn.Module):
    def __init__(self, d_model=256, max_len=33):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe[:, : x.size(1), :]
        return x
    
class MantisEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=6,
                 ffn_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = num_heads,
            dim_feedforward = ffn_dim,
            dropout = dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)
    
    
if __name__ == "__main__":
    dummy_patches = torch.randn(8, 32, 256)
    cls_pos = ClsAndPositional(d_model=256)
    enc = MantisEncoder()
    
    tokens = cls_pos(dummy_patches)
    print(tokens.shape)
    
    out = enc(tokens)
    print(out.shape)
    
    cls_embedding = out[:, 0, :]
    print(cls_embedding.shape)