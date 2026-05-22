# 34241 Digital Video Technology Project

## Impact of Film Restoration on Compression Efficiency and Visual Quality

This repository contains the implementation, evaluation scripts, and video sequences used for the course project in **34241 Digital Video Technology**.

The project investigates how restoration techniques influence both compression efficiency and visual quality for degraded archival film material using HEVC/x265 encoding.

---

## Repository Contents

### Python Scripts

- `restoration.py`  
  Main restoration pipeline implemented using Python and OpenCV.  
  Includes:
  - grain reduction
  - scratch reduction
  - contrast enhancement
  - color correction
  - saturation adjustment
  - sharpening

- `blur_and_temp_calc.py`  
  Script used for computing:
  - blur measure
  - temporal consistency

  using OpenCV and `scikit-image`.

---

## Video Sequences

- `sequences_original_30_sec.zip`  
  Original extracted archival video clips.

- `sequences_restored_mp4.zip`  
  Restored video sequences generated using the restoration framework.

- `sequences_encoded.zip`  
  HEVC/x265 encoded versions used for compression evaluation.

---

## Dataset

The archival material was obtained from the Swedish Film Institute:

- Göteborg (1973)  
  https://www.filmarkivet.se/movies/goteborg/

- Alla Mina Små Holmars Ära (1974)  
  https://www.filmarkivet.se/movies/alla-sma-holmars-ara/

---

## Encoding Configuration

All sequences were encoded using FFmpeg together with the x265 HEVC encoder.

Encoding settings:
- Codec: H.265 / HEVC (libx265)
- Preset: medium
- CRF: 28
- Pixel format: yuv420p
- Container format: MP4

Example command:

```bash
ffmpeg -i input.mp4 -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p -tag:v hvc1 output.mp4
