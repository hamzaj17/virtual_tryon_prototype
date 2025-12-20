# Virtual Try-On Prototype (CatVTON-based)

This is a **local prototype**:
- User uploads a **person photo**
- User selects a **garment from a local catalogue**
- App generates a **try-on output** and saves everything to disk (inputs, auto-mask, result)

> No external datasets are used. Only pretrained model weights are downloaded automatically at first run.

## 1) Requirements
- Python 3.10+ (3.10/3.11 recommended)
- NVIDIA GPU strongly recommended (CUDA). CPU will work but can be **very slow**.

## 2) Setup

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The UI will open in your browser.

## 3) Add garments to the catalogue
Put garment images in:
```
assets/garments/
```

Expected format:
- Prefer **PNG with transparent background**
- Name them nicely, e.g. `blue_shirt.png`, `red_dress.png`

Then run:
```bash
python scripts/rebuild_catalog.py
```

## 4) Where files are saved
- Uploaded people images: `data/uploads/people/`
- Generated sessions: `data/outputs/<session_id>/`
  - `person.png`
  - `garment.png`
  - `auto_mask.png`
  - `mask_overlay.png`
  - `result.png`

## Notes / Credits
This prototype uses the CatVTON diffusion try-on approach and loads CatVTON attention checkpoints + a Stable Diffusion inpainting base model.
Please check the upstream licenses before any use beyond a prototype.
