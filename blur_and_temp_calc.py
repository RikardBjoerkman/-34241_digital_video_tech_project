import cv2
import numpy as np
from skimage.measure import blur_effect


VIDEOS = {
    "gothenburg_original": "gothenburg_30s.mp4",
    "gothenburg_mild": "gothenburg_1973_mild_restored.mp4",
    "gothenburg_strong": "gothenburg_1973_strong_restored.mp4",

    "alla_mina_original": "alla_sma_halmars_ara_30s.mp4",
    "alla_mina_mild": "alla_sma_halmars_ara_1974_mild_restored.mp4",
    "alla_mina_strong": "alla_sma_halmars_ara_1974_strong_restored.mp4",
}


def compute_metrics(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    blur_values = []
    temporal_values = []

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for skimage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Blur measure: 0 = sharp, 1 = blurred
        blur = blur_effect(rgb, h_size=11, channel_axis=2)
        blur_values.append(blur)

        # Temporal consistency: mean absolute difference between consecutive grayscale frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            sad = np.mean(np.abs(gray - prev_gray))
            temporal_values.append(sad)

        prev_gray = gray

    cap.release()

    return {
        "blur_mean": np.mean(blur_values),
        "blur_std": np.std(blur_values),
        "temporal_mean": np.mean(temporal_values),
        "temporal_std": np.std(temporal_values),
    }


if __name__ == "__main__":
    for name, path in VIDEOS.items():
        metrics = compute_metrics(path)

        print(f"\n{name}")
        print(f"Blur mean: {metrics['blur_mean']:.4f}")
        print(f"Blur std: {metrics['blur_std']:.4f}")
        print(f"Temporal consistency mean: {metrics['temporal_mean']:.4f}")
        print(f"Temporal consistency std: {metrics['temporal_std']:.4f}")