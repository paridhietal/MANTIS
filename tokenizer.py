import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamTokenizer(nn.Module):
    def __init__(self, d_model=256, num_patches=32, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pool = nn.AdaptiveAvgPool1d(num_patches)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
    
class MantisTokenizer(nn.Module):
    def __init__(self, d_model=256, num_patches=32, kernel_size=7):
        super().__init__()
        self.raw_stream = StreamTokenizer(d_model, num_patches, kernel_size)
        self.diff_stream = StreamTokenizer(d_model, num_patches, kernel_size)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        diff = x[:, 1:] - x[:, :-1]
        diff = F.pad(diff, (1, 0), value=0.0)
        
        raw_tokens = self.raw_stream(x)
        diff_tokens = self.diff_stream(diff)
        
        fused = torch.cat([raw_tokens, diff_tokens], dim=-1)
        fused = self.fuse(fused)
        fused = self.fuse_norm(fused)
        return fused
    
if __name__ == "__main__":
    tok = MantisTokenizer(d_model=256, num_patches=32)
    dummy = torch.randn(8, 512)
    out = tok(dummy)
    print(out.shape)
        