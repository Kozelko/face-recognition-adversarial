"""
PAD (Presentation Attack Detection) modul — MiniFASNetV2 (Silent-Face-Anti-Spoofing).

Slúži na overenie Tézy C/D: či fyzický/hybridný adversariálny nosič spustí
liveness detektor. Váhy: models/checkpoints/2.7_80x80_MiniFASNetV2.pth
(Minivision, MIT licencia; architektúra v models/minifasnet_arch.py).

Konvencia tried: 0 = fake(print), 1 = live (reálna tvár), 2 = fake(replay).
`liveness` = softmax pravdepodobnosť triedy 1 (overené: reálna tvár -> live 0.81,
print-proxy -> live 0.06, čiže model diskriminuje).

POZOR na škálovanie: Minivision checkpointy očakávajú vstup v rozsahu [0, 255]
(NE [-1,1], NIE [0,1]) — ich transform robí len numpy->tensor bez normalizácie.
Tento modul preto konvertuje vstup: tensor v [-1,1] (konvencia zvyšku repa) sa
prepočíta na [0,255]; numpy RGB HWC uint8 ostáva [0,255].
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.minifasnet_arch import MiniFASNetV2

_WEIGHTS = os.path.join(os.path.dirname(__file__), "checkpoints", "2.7_80x80_MiniFASNetV2.pth")
_LIVE_CLASS = 1


class MiniFASNetPAD(nn.Module):
    """MiniFASNetV2 ako liveness klasifikátor. `forward` vracia live-prob v [0,1]."""

    def __init__(self, device: str = "cpu", crop_size: int = 80):
        super().__init__()
        self.device = torch.device(device)
        self.crop_size = int(crop_size)
        kernel = ((self.crop_size + 15) // 16, (self.crop_size + 15) // 16)
        net = MiniFASNetV2(embedding_size=128, conv6_kernel=kernel)
        static = torch.load(_WEIGHTS, map_location="cpu")
        # Váhy sú uložené ako DataParallel -> odstráň "module." prefix.
        static = {k.replace("module.", "", 1): v for k, v in static.items()}
        net.load_state_dict(static)
        net.eval()
        # Detektor je zmrazený: gradient tečie len do vstupu (PAD-aware loss
        # v ensemble_hybrid_patch), nie do váh.
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(self.device)

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
            if x.dim() == 3 and x.shape[-1] == 3:            # HWC uint8 RGB (0..255) -> CHW [0,255]
                x = x.permute(2, 0, 1).float()
        if not torch.is_tensor(x):
            raise TypeError("MiniFASNetPAD: vstup musí byť torch.Tensor alebo numpy RGB.")
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float()
        # Repo konvencia vstupu je [-1,1]; Minivision očakáva [0,255].
        if float(x.min()) < -0.01:
            x = (x + 1.0) * 127.5
        elif float(x.max()) <= 1.5:
            x = x * 255.0
        return x.to(self.device)

    def forward(self, x) -> torch.Tensor:
        """Live-prob triedy 1. BEZ @torch.no_grad() — diferencovateľné voči
        vstupu, aby PAD-aware penalizácia v útoku naozaj dostávala gradient.
        Váhy sú zmrazené (requires_grad_(False))."""
        x = self._to_tensor(x)
        x = F.interpolate(x, size=(self.crop_size, self.crop_size),
                          mode="bilinear", align_corners=False)
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, _LIVE_CLASS]


def load_pad_model(device: str = "cpu", crop_size: int = 80) -> MiniFASNetPAD:
    return MiniFASNetPAD(device=device, crop_size=crop_size)


if __name__ == "__main__":
    m = load_pad_model()
    print("MiniFASNetPAD loaded OK, live-class index:", _LIVE_CLASS)
