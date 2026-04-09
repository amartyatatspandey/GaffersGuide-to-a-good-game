"""
Run PitchCalibrator on one test sample and compare predicted vs ground-truth homography.
"""
from pathlib import Path

import torch

from backend.app.services.calibration_dataset import SoccerNetCalibrationDataset
from backend.app.services.calibration_model import PitchCalibrator


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent / "data" / "soccernet"
    dataset = SoccerNetCalibrationDataset(root_dir=root_dir, split="test")
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples under {root_dir}. Run soccernet_loader and unzip calibration test split."
        )

    # Get first valid sample (image, label).
    image, label = dataset[0]
    if image is None or label is None:
        raise RuntimeError("First sample is invalid (missing file or homography).")

    model_path = Path(__file__).resolve().parent.parent / "models" / "soccernet_calib_v1.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_calibration.py first.")

    model = PitchCalibrator()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        batch = image.unsqueeze(0)
        pred = model(batch).squeeze(0)

    pred_3x3 = pred.reshape(3, 3)
    label_3x3 = label.reshape(3, 3)

    print("Predicted Homography (3x3):")
    print(pred_3x3)
    print("\nGround Truth Homography (3x3):")
    print(label_3x3)

    mse = torch.nn.functional.mse_loss(pred, label).item()
    print(f"\nMean Squared Error: {mse:.6f}")


if __name__ == "__main__":
    main()
