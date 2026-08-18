# AI Image Analysis (Task 1)

A simple web app that uses AI (CLIP) to identify what's in an image.

## What it does
Upload a photo (or paste an image URL), and the AI compares it against a
list of ~280 possible labels (animals, dinosaurs, food, vehicles, birds,
objects, etc.) and shows the top 5 best matches with confidence scores.

## How it works
- **Model:** CLIP (`ViT-B/32`) by OpenAI — a general-purpose vision-language
  model, pretrained on a huge dataset of image-text pairs.
- **Why CLIP instead of a fixed-class detector (like YOLO):** YOLO can only
  recognize the ~80 categories it was trained on. CLIP is "open-vocabulary" —
  it can compare an image against *any* list of text labels you give it,
  which is why it can recognize things like "a dinosaur" or "an anime
  character" that YOLO can't.
- Runs 100% locally on your computer after the first-time model download —
  no API key, no internet needed afterward, no usage limits.

## Setup
```bash
pip install flask torch torchvision pillow requests --break-system-packages
pip install git+https://github.com/openai/CLIP.git --break-system-packages
```

## Folder structure
```
task1_web/
├── app.py
├── README.md
├── templates/
│   └── index.html
└── static/
    └── uploads/
```

## Run it
```bash
python3 app.py
```
Then open **http://127.0.0.1:8000** in your browser.

## Usage
- Upload an image file, OR paste a direct image URL
- Click "Analyze Image"
- See the top 5 matching labels with confidence percentages

## Known limitations
- Works best on real-world photos with a clear, unobstructed subject.
- Can't reliably tell apart very similar-looking things (e.g. phone/laptop
  brands) since it identifies by appearance, not logos or fine detail.
- Confidence scores are relative to the label list in `LABELS` (in
  `app.py`) — it always picks its best guess from that list, so results
  are only as good as the labels provided.
- Some websites block direct image URL fetching (bot protection) — file
  upload is the more reliable option.

## Credits
- Model: [OpenAI CLIP](https://github.com/openai/CLIP)
- Built with Flask, PyTorch, and Pillow
