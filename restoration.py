import cv2
import numpy as np

PIPELINES = {
    "goteborg": {
        "grain_reduction": True,
        "scratch_reduction": True,
        "contrast": True,
        "color_correction": True,
        "saturation": True,
        "sharpen": True,
    },
    "alla_mina": {
        "grain_reduction": True,
        "scratch_reduction": True,
        "contrast": False,
        "color_correction": True,
        "saturation": True,
        "sharpen": True,
    },
}


def reduce_grain_luma_strong(frame):
    """Stronger film grain reduction on the luminance channel."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_reduced = cv2.fastNlMeansDenoising(
        y,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    restored = cv2.merge((y_reduced, cr, cb))
    return cv2.cvtColor(restored, cv2.COLOR_YCrCb2BGR)


def reduce_scratches_luma_strong(frame):
    """Stronger scratch reduction using median filtering on luminance."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_filtered = cv2.medianBlur(y, 5)

    restored = cv2.merge((y_filtered, cr, cb))
    return cv2.cvtColor(restored, cv2.COLOR_YCrCb2BGR)


def strong_contrast(frame, alpha=1.18, beta=8):
    """Stronger global contrast/brightness adjustment."""
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def gray_world_white_balance(frame):
    """Simple gray-world white balance correction."""
    result = frame.astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    eps = 1e-6

    result[:, :, 0] *= avg_gray / (avg_b + eps)
    result[:, :, 1] *= avg_gray / (avg_g + eps)
    result[:, :, 2] *= avg_gray / (avg_r + eps)

    return np.clip(result, 0, 255).astype(np.uint8)


def increase_saturation(frame, saturation_scale=1.20):
    """Mild saturation enhancement in HSV colorspace to restore chroma."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen(frame):
    """Mild sharpening after stronger filtering."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    sharp = cv2.addWeighted(frame, 1.35, blurred, -0.35, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_frame(frame, settings):
    if settings["grain_reduction"]:
        frame = reduce_grain_luma_strong(frame)

    if settings["scratch_reduction"]:
        frame = reduce_scratches_luma_strong(frame)

    if settings["contrast"]:
        frame = strong_contrast(frame)

    if settings["color_correction"]:
        frame = gray_world_white_balance(frame)

    if settings.get("saturation", False):
        frame = increase_saturation(frame)

    if settings["sharpen"]:
        frame = sharpen(frame)

    return frame


def process_video(input_path, output_path, pipeline_name):
    settings = PIPELINES[pipeline_name]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        restored = process_frame(frame, settings)
        out.write(restored)

        current += 1
        if current % 100 == 0:
            print(f"Processed {current}/{frame_count} frames")

    cap.release()
    out.release()

    print(f"Done. Restored video saved as: {output_path}")


if __name__ == "__main__":
    process_video(
        input_path="gothenburg_30s.mp4",
        output_path="gothenburg_1973_strong_restored.avi",
        pipeline_name="goteborg"
    )

    process_video(
        input_path="alla_sma_halmars_ara_30s.mp4",
        output_path="alla_sma_halmars_ara_1974_strong_restored.avi",
        pipeline_name="alla_mina"
    )