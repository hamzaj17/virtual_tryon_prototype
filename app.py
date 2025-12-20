import os
import gradio as gr
from PIL import Image

from core.config import AppConfig, ModelConfig
from core.catalog import load_catalog, rebuild_catalog
from core.tryon_engine import TryOnEngine

app_cfg = AppConfig()
model_cfg = ModelConfig()

# Ensure folders exist
os.makedirs(app_cfg.garments_dir, exist_ok=True)
os.makedirs(app_cfg.uploads_dir, exist_ok=True)
os.makedirs(app_cfg.outputs_dir, exist_ok=True)

# Build catalog if missing
if not os.path.exists(app_cfg.catalog_json):
    rebuild_catalog(app_cfg.garments_dir, app_cfg.catalog_json)

catalog = load_catalog(app_cfg.catalog_json)

engine = None

def _get_engine():
    global engine
    if engine is None:
        engine = TryOnEngine(model_cfg, app_cfg)
    return engine

def refresh_catalog():
    rebuild_catalog(app_cfg.garments_dir, app_cfg.catalog_json)
    global catalog
    catalog = load_catalog(app_cfg.catalog_json)
    return gr.Dropdown(choices=[g.id for g in catalog], value=(catalog[0].id if catalog else None))

def garment_preview(garment_id):
    g = next((x for x in catalog if x.id == garment_id), None)
    if g is None:
        return None
    return Image.open(g.path).convert("RGB")

def run_tryon(person_img, garment_id, steps, guidance):
    if person_img is None:
        raise gr.Error("Please upload a person photo.")
    g = next((x for x in catalog if x.id == garment_id), None)
    if g is None:
        raise gr.Error("Please select a garment (or refresh the catalogue).")

    eng = _get_engine()
    result, out_img, overlay, mask = eng.run(person_img, g, steps=steps, guidance=guidance)

    info = (
        f"Session: {result.session_id}\n"
        f"Saved to: {os.path.join(app_cfg.outputs_dir, result.session_id)}\n"
        f"Files: result.png, person.png, garment.png, auto_mask.png, mask_overlay.png"
    )
    return out_img, overlay, mask, info

with gr.Blocks(title="Virtual Try-On Prototype") as demo:
    gr.Markdown(
        """# Virtual Try-On (Prototype)

Upload a **person photo**, select a garment from **assets/garments/**, then click **Generate**.

Tip: For best results use:
- front-facing photo
- arms visible
- simple background
- garment images with clean background (PNG preferred)
"""
    )

    with gr.Row():
        person_in = gr.Image(type="pil", label="Person photo (upload)", height=360)
        with gr.Column():
            garment_id = gr.Dropdown(
                choices=[g.id for g in catalog],
                value=(catalog[0].id if catalog else None),
                label="Garment (from local catalogue)",
            )
            garment_img = gr.Image(type="pil", label="Garment preview", height=240)
            refresh_btn = gr.Button("Refresh catalogue (re-scan assets/garments)")

    with gr.Row():
        steps = gr.Slider(15, 60, value=35, step=1, label="Inference steps (quality vs speed)")
        guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Guidance scale")

    gen_btn = gr.Button("Generate Try-On", variant="primary")

    with gr.Row():
        out_img = gr.Image(type="pil", label="Try-on result", height=360)
        overlay_img = gr.Image(type="pil", label="Auto-mask overlay (debug)", height=360)

    with gr.Row():
        mask_img = gr.Image(type="pil", label="Auto-mask (debug)", height=360)
        info = gr.Textbox(label="Run info", lines=5)

    garment_id.change(fn=garment_preview, inputs=garment_id, outputs=garment_img)
    refresh_btn.click(fn=refresh_catalog, inputs=None, outputs=garment_id)
    gen_btn.click(fn=run_tryon, inputs=[person_in, garment_id, steps, guidance], outputs=[out_img, overlay_img, mask_img, info])

demo.launch()
