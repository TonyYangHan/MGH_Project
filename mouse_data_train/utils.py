# utils_pt_min.py
import torch, numpy as np
import torch.nn as nn

# 2×64 ELU -> 1 logit (binary)
class MLP64(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(),
            nn.Linear(64, 64),        nn.ELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)  # logits

def load_model_state(path: str, input_dim: int, device: str | None = None) -> nn.Module:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # strip DataParallel 'module.' if present
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = MLP64(input_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    device = next(model.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = []
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i+batch_size])
        out.append(torch.sigmoid(logits))
    return torch.cat(out).cpu().numpy()  # shape (N,)