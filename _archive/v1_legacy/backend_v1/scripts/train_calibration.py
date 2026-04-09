"""
Train PitchCalibrator on SoccerNet calibration data (test split).

Saves model state dict to backend/models/soccernet_calib_v1.pt.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.app.services.calibration_dataset import (
    SoccerNetCalibrationDataset,
    _collate_skip_none,
)
from backend.app.services.calibration_model import PitchCalibrator


def get_device() -> torch.device:
    """Select device: MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    root_dir = Path(__file__).resolve().parent.parent / "data" / "soccernet"
    dataset = SoccerNetCalibrationDataset(root_dir=root_dir, split="test")
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples under {root_dir}. Run soccernet_loader and unzip calibration test split."
        )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_skip_none,
    )

    model = PitchCalibrator().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    model.train()
    batch_count = 0
    for epoch in range(epochs):
        for batch_idx, (images, homographies) in enumerate(loader):
            images = images.to(device)
            homographies = homographies.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, homographies)
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Loss (batch {batch_count}): {loss.item():.6f}")

    out_path = Path(__file__).resolve().parent.parent / "models" / "soccernet_calib_v1.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved state dict to {out_path}")


if __name__ == "__main__":
    main()
