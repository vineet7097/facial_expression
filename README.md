# Facial Expression Recognition (ResNet-18)

An interactive deep learning pipeline for real-time and static **facial expression classification** built with PyTorch, OpenCV, and Google Colab. The model classifies human facial expressions across six emotional categories using a fine-tuned **ResNet-18** backbone with automated face isolation and an interactive browser UI.

---

## Supported Expression Classes

| Index | Class | Description |
| :--- | :--- | :--- |
| `0` | **Natural** | Neutral / baseline resting expression |
| `1` | **Anger** | Furrowed brow, tense jaw, or angry affect |
| `2` | **Fear** | Widened eyes, raised eyebrows, fearful affect |
| `3` | **Joy** | Smiling, raised cheeks, positive affect |
| `4` | **Sadness** | Downturned mouth, drooping eyelids, sad affect |
| `5` | **Surprise** | Raised eyebrows, open mouth, startled affect |

---

## Key Features

- **Automated Face Isolation:** Uses OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`) to detect and crop the primary subject face with proportional padding, plus a lower-center portrait fallback for unconstrained photos.
- **Interactive Colab Interface:** Built-in `ipywidgets` controls supporting both **Live Webcam Capture** (via Colab's browser JavaScript bridge) and **Local File Upload**.
- **Visual & Quantitative Output:** Displays the isolated face alongside a ranked horizontal confidence bar chart and a formatted probability breakdown across all six classes.
- **Hardware Agnostic:** Automatically runs on CUDA GPU when available or falls back cleanly to CPU.

---

## Repository Structure

```text
├── facial_exp.ipynb    # Complete 4-cell Google Colab inference & visualization notebook
└── README.md           # Project documentation
