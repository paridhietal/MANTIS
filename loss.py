import torch
import torch.nn as nn  
import torch.nn.functional as F 

def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T 
    sim = sim / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    
    targets = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)
    
    loss = F.cross_entropy(sim, targets)
    return loss

if __name__ == "__main__":
    B, D = 8, 128
    z1 = F.normalize(torch.randn(B, D), dim=-1)
    z2 = F.normalize(torch.randn(B, D), dim=-1)
    loss = nt_xent_loss(z1, z2)
    print(loss.item())