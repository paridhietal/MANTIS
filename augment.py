import torch
import torch.nn.functional as F 

def random_crop_resize(x, min_keep=0.8):
    B, L = x.shape
    out = torch.empty_like(x)
    
    for i in range(B):
        keep_frac = torch.empty(1).uniform_(min_keep, 1.0).item()
        keep_len = max(2, int(L * keep_frac))
        start = torch.randint(0, L - keep_len + 1, (1,)).item()
        
        chunk = x[i, start:start + keep_len]
        chunk = chunk.unsqueeze(0).unsqueeze(0)
        stretched = F.interpolate(chunk, size=L, mode="linear", align_corners=False)
        out[i] = stretched.squeeze(0).squeeze(0)
        
    return out

if __name__ == "__main__":
    dummy = torch.randn(8, 512)
    view1 = random_crop_resize(dummy)
    view2 = random_crop_resize(dummy)
    print(view1.shape, view2.shape)
    