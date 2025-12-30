import os
import time
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw

from backend_bridge import BackendBridge


# =========================
# VISUAL THEME
# =========================
APP_BG        = "#1F2124"
HEADER_BG     = "#15181B"
PANEL_BG      = "#2B2F33"
CARD_BG       = "#2F343A"
BORDER_DIM    = "#3B4047"
TEXT_PRIMARY  = "#E9EEF4"
TEXT_MUTED    = "#A7AFB8"
TEXT_DIM      = "#6F7782"

ACCENT_CYAN   = "#06CBFF"
ACCENT_CYAN_D = "#00A9D6"
ACCENT_CYAN_D2 = "#0285AA"
ACCENT_GREEN  = "#53F63B"

BTN_BG        = "#3A3F45"
BTN_BG_HOVER  = "#454B52"
BTN_BG_DOWN   = "#2E3339"

GEN_BG        = "#06CBFF"
GEN_BG_HOVER  = "#33D6FF"
GEN_BG_DOWN   = "#00A9D6"
GEN_TEXT      = "#0B1418"

DIVIDER_CYAN  = "#1AA7C7"

UPLOAD_ICON_REL = os.path.join("assets", "icons", "add_box_24dp_2854C5_FILL0_wght400_GRAD0_opsz24.png")


# =========================
# UTILS
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_font(root, family="Poppins"):
    try:
        import tkinter.font as tkfont
        families = set(tkfont.families(root))
        if family in families:
            return family
    except Exception:
        pass
    return "Segoe UI"


def rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    items = []
    items.append(canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, style="pieslice", **kwargs))
    items.append(canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, style="pieslice", **kwargs))
    items.append(canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, style="pieslice", **kwargs))
    items.append(canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, style="pieslice", **kwargs))
    items.append(canvas.create_rectangle(x1+r, y1, x2-r, y2, **kwargs))
    items.append(canvas.create_rectangle(x1, y1+r, x2, y2-r, **kwargs))
    return items


def icon_lightning(size=18, color=GEN_TEXT):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = [(10,1),(4,10),(9,10),(6,17),(14,7),(9,7)]
    d.polygon(pts, fill=color)
    return im


def tint_icon_to_cyan(im_rgba, tint_hex=ACCENT_CYAN):
    r_t, g_t, b_t = Image.new("RGBA", (1, 1), tint_hex).getpixel((0, 0))[:3]
    im = im_rgba.convert("RGBA")
    px = im.load()
    w, h = im.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > 0:
                px[x, y] = (r_t, g_t, b_t, a)
    return im


def center_crop_to_aspect(im, target_w, target_h):
    """Center-crop image to match target aspect ratio so it fills the stage."""
    if target_w <= 1 or target_h <= 1:
        return im
    im = im.copy()
    w, h = im.size
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return im.crop((x1, 0, x1 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        return im.crop((0, y1, w, y1 + new_h))


# =========================
# CUSTOM NEON BUTTON
# =========================
class NeonButton(tk.Frame):
    def __init__(self, parent, text, command=None, *,
                 w=240, h=44,
                 bg=BTN_BG, hover=BTN_BG_HOVER, down=BTN_BG_DOWN,
                 fg=TEXT_PRIMARY, font=("Segoe UI", 11, "bold"),
                 radius=10,
                 icon_pil=None,
                 padx=14,
                 glow=False,
                 glow_color=ACCENT_CYAN_D2,
                 chevron=False,
                 top_highlight=True,
                 border=True):
        super().__init__(parent, bg=parent["bg"])
        self.command = command
        self.w, self.h = w, h
        self.bg = bg
        self.hover = hover
        self.down = down
        self.fg = fg
        self.font = font
        self.radius = radius
        self.padx = padx
        self.glow = glow
        self.glow_color = glow_color
        self.chevron = chevron
        self.top_highlight = top_highlight
        self.border = border
        self._state = "normal"

        self.canvas = tk.Canvas(self, width=w, height=h, bg=self["bg"], highlightthickness=0)
        self.canvas.pack()

        self.icon_img = ImageTk.PhotoImage(icon_pil) if icon_pil is not None else None
        self.text = text
        self._draw(self.bg)

        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

    def set_disabled(self, disabled=True):
        self._state = "disabled" if disabled else "normal"
        self._draw(self.bg if not disabled else "#2A2E33")

    def _draw(self, fill):
        self.canvas.delete("all")

        if self.glow:
            rounded_rect(
                self.canvas,
                2, 2, self.w - 2, self.h - 2,
                self.radius + 2,
                fill=self.glow_color,
                outline=self.glow_color,
                width=1
            )

        outline = BORDER_DIM if self.border else fill
        width = 1 if self.border else 0

        rounded_rect(
            self.canvas,
            6, 6, self.w - 6, self.h - 6,
            self.radius,
            fill=fill,
            outline=outline,
            width=width
        )

        if self.top_highlight:
            self.canvas.create_line(16, 14, self.w - 16, 14, fill="#FFFFFF", width=1)

        x = self.padx + 6
        if self.icon_img is not None:
            self.canvas.create_image(x + 10, self.h//2, image=self.icon_img)
            x += 30

        self.canvas.create_text(x + 4, self.h//2, text=self.text, fill=self.fg,
                                font=self.font, anchor="w")

        if self.chevron:
            cx = self.w - 26
            cy = self.h // 2
            self.canvas.create_polygon(
                (cx - 6, cy - 8, cx + 6, cy, cx - 6, cy + 8),
                fill=self.fg, outline=""
            )

    def _inside(self, event):
        return 0 <= event.x <= self.w and 0 <= event.y <= self.h

    def _on_enter(self, _e):
        if self._state == "disabled": return
        self._draw(self.hover)

    def _on_leave(self, _e):
        if self._state == "disabled": return
        self._draw(self.bg)

    def _on_down(self, _e):
        if self._state == "disabled": return
        self._draw(self.down)

    def _on_up(self, e):
        if self._state == "disabled": return
        self._draw(self.hover if self._inside(e) else self.bg)
        if self._inside(e) and self.command:
            self.command()


# =========================
# CUSTOM NEON SLIDER
# =========================
class NeonSlider(tk.Frame):
    def __init__(self, parent, label, from_, to_, initial, on_change=None, *,
                 width=320, track_h=6, knob_r=9):
        super().__init__(parent, bg=parent["bg"])
        self.label = label
        self.from_ = float(from_)
        self.to_ = float(to_)
        self.value = float(initial)
        self.on_change = on_change
        self.width = width
        self.track_h = track_h
        self.knob_r = knob_r

        root = parent.winfo_toplevel()
        family = getattr(root, "font_family", "Segoe UI")

        top = tk.Frame(self, bg=self["bg"])
        top.pack(fill="x")
        self.lbl = tk.Label(top, text=label, bg=self["bg"], fg=TEXT_MUTED, font=(family, 11))
        self.lbl.pack(side="left")

        self.val_lbl = tk.Label(top, text=f"{self.value:.2f}" if self.to_ <= 2 else f"{int(self.value)}",
                                bg=self["bg"], fg=TEXT_DIM, font=(family, 10))
        self.val_lbl.pack(side="right")

        self.c = tk.Canvas(self, width=width, height=28, bg=self["bg"], highlightthickness=0)
        self.c.pack(pady=(6, 0))

        self._dragging = False
        self.c.bind("<ButtonPress-1>", self._down)
        self.c.bind("<B1-Motion>", self._move)
        self.c.bind("<ButtonRelease-1>", self._up)

        self._render()

    def _pos_from_value(self):
        t = (self.value - self.from_) / (self.to_ - self.from_)
        x0 = 10
        x1 = self.width - 10
        return x0 + t * (x1 - x0)

    def _value_from_pos(self, x):
        x0, x1 = 10, self.width - 10
        t = clamp((x - x0) / (x1 - x0), 0, 1)
        return self.from_ + t * (self.to_ - self.from_)

    def _render(self):
        self.c.delete("all")
        x0, x1 = 10, self.width - 10
        y = 14
        self.c.create_line(x0, y, x1, y, fill="#444A52", width=self.track_h, capstyle="round")
        px = self._pos_from_value()
        self.c.create_line(x0, y, px, y, fill=ACCENT_CYAN, width=self.track_h, capstyle="round")
        self.c.create_oval(px - self.knob_r, y - self.knob_r, px + self.knob_r, y + self.knob_r,
                           fill="#FFFFFF", outline="#D8DEE6", width=1)

        if self.to_ <= 2:
            self.val_lbl.config(text=f"{self.value:.2f}")
        else:
            self.val_lbl.config(text=f"{int(round(self.value))}")

    def _down(self, e):
        self._dragging = True
        self._apply_pos(e.x)

    def _move(self, e):
        if not self._dragging: return
        self._apply_pos(e.x)

    def _up(self, _e):
        self._dragging = False

    def _apply_pos(self, x):
        v = self._value_from_pos(x)
        self.value = v
        self._render()
        if self.on_change:
            self.on_change(v)


# =========================
# SCROLLABLE FRAME
# =========================
class ScrollFrame(tk.Frame):
    def __init__(self, parent, bg=CARD_BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfigure(self.win, width=e.width)

    def _on_wheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")


# =========================
# IMAGE HELPERS
# =========================
def fit_into(im, box):
    w, h = im.size
    bw, bh = box
    s = min(bw / w, bh / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return im.resize((nw, nh), Image.LANCZOS)


def thumb_tile(im_rgba, size=72):
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=12, fill="#E9EDF3", outline="#D7DEE8", width=1)

    content = im_rgba.copy().convert("RGBA")
    content = fit_into(content, (size - 16, size - 16))
    x = (size - content.size[0]) // 2
    y = (size - content.size[1]) // 2
    tile.alpha_composite(content, (x, y))
    return tile


def mesh_overlay(im_rgba, color=ACCENT_CYAN):
    im = im_rgba.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    line = (*Image.new("RGBA", (1, 1), color).getpixel((0, 0))[:3], 80)
    step = max(24, min(w, h) // 12)
    for x in range(0, w, step):
        d.line((x, 0, x, h), fill=line, width=1)
    for y in range(0, h, step):
        d.line((0, y, w, y), fill=line, width=1)
    return im


def composite_tryon(base_rgba, garment_rgba, fit=1.0, x_off=0, y_off=0, opacity=0.85):
    base = base_rgba.copy().convert("RGBA")
    g = garment_rgba.copy().convert("RGBA")

    bw, bh = base.size
    target = int(min(bw, bh) * 0.55 * fit)
    g.thumbnail((target, target), Image.LANCZOS)

    alpha = g.split()[-1]
    alpha = alpha.point(lambda a: int(a * opacity))
    g.putalpha(alpha)

    gx, gy = g.size
    x = bw // 2 - gx // 2 + int(x_off)
    y = bh // 2 - gy // 2 + int(y_off)
    base.alpha_composite(g, (x, y))
    return base


def garment_mask_preview(garment_rgba):
    g = garment_rgba.convert("RGBA")
    bg = Image.new("RGBA", g.size, (0, 0, 0, 255))
    bg.alpha_composite(g, (0, 0))
    return bg


# =========================
# MAIN APP
# =========================
class TryOnStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Try-On Studio")
        self.geometry("1200x780")
        self.minsize(1100, 720)
        self.configure(bg=APP_BG)

        self.font_family = get_font(self, "Poppins")
        self.font_title = (self.font_family, 24, "bold")

        self.q = queue.Queue()

        # backend
        self.backend = None
        try:
            self.backend = BackendBridge()
        except Exception as e:
            print("BackendBridge init failed:", e)

        self.subject_img = None
        self.garment_img = None
        self.selected_catalog = 0
        self.result_img = None

        self.backend_debug = None  # (overlay_rgb, mask_rgb, info)

        self.fit = 1.00
        self.x_off = 0
        self.opacity = 0.85

        self._build_ui()
        self._load_garments_from_assets()
        self._set_status_ready(True)

        self.after(100, self._poll_queue)

        if self.font_family != "Poppins":
            self.after(600, self._warn_font_missing)

    def _warn_font_missing(self):
        messagebox.showwarning(
            "Poppins not found",
            "Poppins is not installed.\n\n"
            "UI is using a fallback font.\n"
            "Install Poppins to match your reference design exactly."
        )

    # -------------------------
    # UI STRUCTURE
    # -------------------------
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_header()
        self._build_content()
        self._build_bottom_bar()

    def _build_header(self):
        header = tk.Frame(self, bg=HEADER_BG, height=84)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        left = tk.Frame(header, bg=HEADER_BG)
        left.pack(side="left", padx=28, pady=18)

        line1 = tk.Frame(left, bg=HEADER_BG)
        line1.pack(anchor="w")
        tk.Label(line1, text="VIRTUAL ", fg=TEXT_PRIMARY, bg=HEADER_BG, font=self.font_title).pack(side="left")
        tk.Label(line1, text="TRY-ON", fg=ACCENT_CYAN, bg=HEADER_BG, font=self.font_title).pack(side="left")
        tk.Label(left, text="STUDIO", fg=ACCENT_CYAN, bg=HEADER_BG, font=self.font_title).pack(anchor="w")

    def _build_content(self):
        content = tk.Frame(self, bg=PANEL_BG)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(2, weight=2)
        content.grid_rowconfigure(0, weight=1)

        self.left = tk.Frame(content, bg=PANEL_BG)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(34, 20), pady=22)

        divider = tk.Frame(content, bg=DIVIDER_CYAN, width=2)
        divider.grid(row=0, column=1, sticky="ns", pady=28, padx=2)

        self.right = tk.Frame(content, bg=PANEL_BG)
        self.right.grid(row=0, column=2, sticky="nsew", padx=(20, 34), pady=22)
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, fg=TEXT_DIM, bg=parent["bg"],
                 font=(self.font_family, 11, "bold")).pack(anchor="w", pady=(0, 10))

    # -------------------------
    # LEFT PANEL
    # -------------------------
    def _load_upload_icon(self, size=18):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, UPLOAD_ICON_REL)
        if os.path.exists(icon_path):
            im = Image.open(icon_path).convert("RGBA")
            im = im.resize((size, size), Image.LANCZOS)
            im = tint_icon_to_cyan(im, ACCENT_CYAN)
            return im
        return None

    def _build_left_panel(self):
        self._section_label(self.left, "STEP 1: UPLOAD PHOTO")

        upload_shell = tk.Frame(self.left, bg=BORDER_DIM)
        upload_shell.pack(fill="x", pady=(0, 18))
        upload_inner = tk.Frame(upload_shell, bg=CARD_BG)
        upload_inner.pack(fill="both", expand=True, padx=1, pady=1)

        accent_bar = tk.Frame(upload_inner, bg=ACCENT_CYAN, width=3)
        accent_bar.pack(side="left", fill="y")

        content = tk.Frame(upload_inner, bg=CARD_BG)
        content.pack(side="left", fill="both", expand=True, padx=14, pady=14)

        upload_icon = self._load_upload_icon(18)

        self.btn_upload = NeonButton(
            content,
            text="Select Subject Photo",
            command=self._on_upload,
            w=380, h=46,
            bg=BTN_BG, hover=BTN_BG_HOVER, down=BTN_BG_DOWN,
            fg=TEXT_PRIMARY,
            font=(self.font_family, 11, "bold"),
            icon_pil=upload_icon,
            top_highlight=False,
            border=True
        )
        self.btn_upload.pack(anchor="w")

        self._section_label(self.left, "STEP 2: BROWSE CATALOGUE")

        cat_outer = tk.Frame(self.left, bg=ACCENT_CYAN)
        cat_outer.pack(fill="both", expand=True, pady=(0, 18))
        cat_inner = tk.Frame(cat_outer, bg=CARD_BG)
        cat_inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.catalog = ScrollFrame(cat_inner, bg=CARD_BG)
        self.catalog.pack(fill="both", expand=True, padx=10, pady=10)

        self._section_label(self.left, "ADJUSTMENTS")

        self.slider_fit = NeonSlider(self.left, "Fit", 0.6, 1.4, self.fit, on_change=self._on_fit_change, width=380)
        self.slider_fit.pack(fill="x", pady=(6, 8))

        tk.Label(self.left, text="Fit Accuracy", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=(self.font_family, 11)).pack(anchor="w", pady=(6, 0))
        tk.Label(self.left, text="Draping Intensity", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=(self.font_family, 11)).pack(anchor="w", pady=(6, 0))

        self.slider_lush = NeonSlider(self.left, "Lush", -120, 120, self.x_off, on_change=self._on_x_change, width=380)
        self.slider_lush.pack(fill="x", pady=(10, 8))

        self.slider_op = NeonSlider(self.left, "Opacity", 0, 100, int(self.opacity * 100),
                                    on_change=self._on_op_change, width=380)
        self.slider_op.pack(fill="x", pady=(10, 0))

    # -------------------------
    # RIGHT PANEL
    # -------------------------
    def _build_right_panel(self):
        tk.Label(self.right, text="THE STUDIO STAGE", fg=TEXT_DIM, bg=PANEL_BG,
                 font=(self.font_family, 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        glow_outer = tk.Frame(self.right, bg=ACCENT_CYAN_D2)
        glow_outer.grid(row=1, column=0, sticky="nsew")
        glow_outer.grid_rowconfigure(0, weight=1)
        glow_outer.grid_columnconfigure(0, weight=1)

        glow_mid = tk.Frame(glow_outer, bg=ACCENT_CYAN_D)
        glow_mid.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        glow_mid.grid_rowconfigure(0, weight=1)
        glow_mid.grid_columnconfigure(0, weight=1)

        stage_border = tk.Frame(glow_mid, bg=ACCENT_CYAN)
        stage_border.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        stage_border.grid_rowconfigure(0, weight=1)
        stage_border.grid_columnconfigure(0, weight=1)

        stage = tk.Frame(stage_border, bg="#0B0D10")
        stage.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        stage.grid_rowconfigure(0, weight=1)
        stage.grid_columnconfigure(0, weight=1)

        self.stage_canvas = tk.Canvas(stage, bg="#0B0D10", highlightthickness=0)
        self.stage_canvas.grid(row=0, column=0, sticky="nsew")

        self.stage_label = tk.Label(stage, text="STUDIO PREVIEW AREA", fg=TEXT_MUTED, bg="#0B0D10",
                                    font=(self.font_family, 11, "bold"))
        self.stage_label.place(relx=0.5, rely=0.92, anchor="center")

        bottom = tk.Frame(self.right, bg=PANEL_BG)
        bottom.grid(row=2, column=0, sticky="ew", pady=(18, 0))

        tk.Label(bottom, text="DEBUG PREVIEWS", fg=TEXT_DIM, bg=PANEL_BG,
                 font=(self.font_family, 11, "bold")).grid(row=0, column=0, sticky="w", columnspan=3, pady=(0, 10))

        self.debug_tiles = []
        for i, name in enumerate(["Original Photo", "Garment Mask", "Mesh Overlay"]):
            tile = self._make_debug_tile(bottom, name)
            tile.grid(row=1, column=i, sticky="w", padx=(0 if i == 0 else 18, 0))
            self.debug_tiles.append(tile)

        self.stage_canvas.bind("<Configure>", lambda e: self._render_stage())

    def _make_debug_tile(self, parent, caption):
        wrapper = tk.Frame(parent, bg=PANEL_BG)
        border_color = ACCENT_CYAN if caption in {"Original Photo", "Garment Mask", "Mesh Overlay"} else BORDER_DIM
        inner_border = tk.Frame(wrapper, bg=border_color)
        inner_border.pack()
        inner = tk.Frame(inner_border, bg="#0B0D10")
        inner.pack(padx=1, pady=1)

        canvas = tk.Canvas(inner, width=84, height=84, bg="#0B0D10", highlightthickness=0)
        canvas.pack()

        lbl = tk.Label(wrapper, text=caption, fg=TEXT_MUTED, bg=PANEL_BG,
                       font=(self.font_family, 10))
        lbl.pack(anchor="w", pady=(6, 0))

        wrapper.canvas = canvas
        return wrapper

    # -------------------------
    # BOTTOM BAR
    # -------------------------
    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=HEADER_BG, height=84)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)

        left = tk.Frame(bar, bg=HEADER_BG)
        left.pack(side="left", padx=28)

        self.dot = tk.Canvas(left, width=14, height=14, bg=HEADER_BG, highlightthickness=0)
        self.dot.pack(side="left")
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill=ACCENT_GREEN, outline=ACCENT_GREEN)

        self.status_lbl = tk.Label(left, text="AI ENGINE READY", fg=ACCENT_GREEN, bg=HEADER_BG,
                                   font=(self.font_family, 12, "bold"))
        self.status_lbl.pack(side="left", padx=(10, 0))

        right = tk.Frame(bar, bg=HEADER_BG)
        right.pack(side="right", padx=28)

        bolt = icon_lightning(18, color=GEN_TEXT)
        self.btn_generate = NeonButton(
            right,
            text="GENERATE TRY-ON",
            command=self._on_generate,
            w=380, h=58,
            bg=GEN_BG, hover=GEN_BG_HOVER, down=GEN_BG_DOWN,
            fg=GEN_TEXT,
            font=(self.font_family, 13, "bold"),
            icon_pil=bolt,
            glow=True,
            glow_color=ACCENT_CYAN_D2,
            chevron=True,
            top_highlight=False,
            border=False
        )
        self.btn_generate.pack()

    # -------------------------
    # CATALOGUE (backend-connected)
    # -------------------------
    def _load_garments_from_assets(self):
        if self.backend is not None:
            try:
                self.backend.refresh_catalog()
                items = self.backend.list_catalog_items()
                self.catalog_items = items  # name/path/image/obj
                if self.catalog_items:
                    self._render_catalog_tiles()
                    return
            except Exception as e:
                print("Backend catalog load failed; fallback to local folder scan:", e)

        # fallback local scan (keeps your behavior)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        garments_dir = os.path.join(base_dir, "assets", "garments")
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        self.catalog_items = []
        files = []
        if os.path.isdir(garments_dir):
            for fn in sorted(os.listdir(garments_dir)):
                ext = os.path.splitext(fn)[1].lower()
                if ext in valid_ext:
                    files.append(os.path.join(garments_dir, fn))

        if not files:
            self._render_empty_catalog_message(garments_dir)
            self.garment_img = None
            self._render_stage()
            self._render_debug()
            return

        for w in self.catalog.inner.winfo_children():
            w.destroy()

        for path in files:
            try:
                im = Image.open(path).convert("RGBA")
                name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").replace("-", " ").title()
                self.catalog_items.append({"name": name, "path": path, "image": im, "obj": None})
            except Exception:
                continue

        if not self.catalog_items:
            self._render_empty_catalog_message(garments_dir)
            self.garment_img = None
            self._render_stage()
            self._render_debug()
            return

        self._render_catalog_tiles()

    def _render_catalog_tiles(self):
        for w in self.catalog.inner.winfo_children():
            w.destroy()

        cols = 5
        pad = 10
        self.tile_refs = []
        self.tile_borders = []

        self.selected_catalog = 0
        self.garment_img = self.catalog_items[self.selected_catalog]["image"]

        for i, item in enumerate(self.catalog_items):
            r, c = divmod(i, cols)

            tile = tk.Frame(self.catalog.inner, bg=CARD_BG)
            tile.grid(row=r, column=c, padx=pad//2, pady=pad//2, sticky="n")

            selected = (i == self.selected_catalog)
            outer_color = ACCENT_CYAN_D2 if selected else BORDER_DIM
            mid_color = ACCENT_CYAN if selected else BORDER_DIM

            glow_outer = tk.Frame(tile, bg=outer_color)
            glow_outer.pack()
            glow_mid = tk.Frame(glow_outer, bg=mid_color)
            glow_mid.pack(padx=1, pady=1)

            inner = tk.Frame(glow_mid, bg="#E9EDF3")
            inner.pack(padx=2, pady=2)

            thumb = thumb_tile(item["image"], size=72)
            img = ImageTk.PhotoImage(thumb)
            self.tile_refs.append(img)

            lbl = tk.Label(inner, image=img, bg="#E9EDF3")
            lbl.pack()

            cap = tk.Label(tile, text=item["name"][:14], fg=TEXT_MUTED, bg=CARD_BG, font=(self.font_family, 9))
            cap.pack(anchor="w", pady=(6, 0))

            def on_click(_e, idx=i):
                self._select_catalog(idx)

            for wdg in (tile, glow_outer, glow_mid, inner, lbl, cap):
                wdg.bind("<Button-1>", on_click)

            self.tile_borders.append((glow_outer, glow_mid))

        self._render_stage()
        self._render_debug()

    def _render_empty_catalog_message(self, garments_dir):
        for w in self.catalog.inner.winfo_children():
            w.destroy()
        msg = tk.Label(
            self.catalog.inner,
            text=f"No garments found.\n\nPut garment images in:\n{garments_dir}\n\n"
                 f"Supported: PNG/JPG/WEBP\n(transparent PNG overlays work best)",
            fg=TEXT_MUTED,
            bg=CARD_BG,
            font=(self.font_family, 10),
            justify="left"
        )
        msg.pack(anchor="w", padx=12, pady=12)

    def _select_catalog(self, idx):
        if not self.catalog_items:
            return
        self.selected_catalog = idx
        self.garment_img = self.catalog_items[idx]["image"]
        self.backend_debug = None

        for i, (outer, mid) in enumerate(self.tile_borders):
            selected = (i == idx)
            outer.configure(bg=ACCENT_CYAN_D2 if selected else BORDER_DIM)
            mid.configure(bg=ACCENT_CYAN if selected else BORDER_DIM)

        self._render_stage()
        self._render_debug()

    # -------------------------
    # EVENTS
    # -------------------------
    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select Subject Photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            im = Image.open(path).convert("RGBA")
            self.subject_img = im
            self.result_img = None
            self.backend_debug = None
            self._render_stage()
            self._render_debug()
        except Exception as e:
            messagebox.showerror("Upload Failed", str(e))

    def _on_fit_change(self, v):
        self.fit = float(v)
        self._render_stage()
        self._render_debug()

    def _on_x_change(self, v):
        self.x_off = float(v)
        self._render_stage()
        self._render_debug()

    def _on_op_change(self, v):
        self.opacity = float(v) / 100.0
        self._render_stage()
        self._render_debug()

    def _set_status_ready(self, ready: bool):
        if ready:
            self.dot.itemconfig(self.dot_id, fill=ACCENT_GREEN, outline=ACCENT_GREEN)
            self.status_lbl.config(text="AI ENGINE READY", fg=ACCENT_GREEN)
        else:
            self.dot.itemconfig(self.dot_id, fill=ACCENT_CYAN, outline=ACCENT_CYAN)
            self.status_lbl.config(text="PROCESSING…", fg=ACCENT_CYAN)

    def _on_generate(self):
        if self.subject_img is None:
            messagebox.showinfo("Missing photo", "Please upload a subject photo first.")
            return
        if self.garment_img is None:
            messagebox.showinfo("Missing garment", "No garment selected (or garments folder is empty).")
            return

        self.btn_generate.set_disabled(True)
        self.btn_upload.set_disabled(True)
        self._set_status_ready(False)

        subj = self.subject_img.copy()
        garment = self.garment_img.copy()
        fit = self.fit
        x_off = self.x_off
        opacity = self.opacity

        def worker():
            try:
                # REAL BACKEND
                if self.backend is not None:
                    item = self.catalog_items[self.selected_catalog]
                    g_obj = item.get("obj", None)

                    if g_obj is not None:
                        steps = 35
                        guidance = 2.5
                        out_rgba, overlay_rgb, mask_rgb, info_text = self.backend.run_tryon(
                            subj, g_obj, steps=steps, guidance=guidance
                        )
                        self.q.put(("ok", out_rgba))
                        self.q.put(("dbg", (overlay_rgb, mask_rgb, info_text)))
                        return

                # PREVIEW MODE (fast)
                time.sleep(0.25)
                result = composite_tryon(subj, garment, fit=fit, x_off=x_off, y_off=0, opacity=opacity)
                self.q.put(("ok", result))

            except Exception as e:
                self.q.put(("err", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                tag, payload = self.q.get_nowait()

                if tag == "ok":
                    self.result_img = payload
                    self._render_stage(final=True)
                    self._render_debug()

                elif tag == "dbg":
                    overlay_rgb, mask_rgb, info_text = payload
                    self.backend_debug = (overlay_rgb, mask_rgb, info_text)
                    print(info_text)
                    self._render_debug()

                else:
                    messagebox.showerror("Generation Error", payload)

                self.btn_generate.set_disabled(False)
                self.btn_upload.set_disabled(False)
                self._set_status_ready(True)

        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # -------------------------
    # RENDERING
    # -------------------------
    def _render_stage(self, final=False):
        cw = max(1, self.stage_canvas.winfo_width())
        ch = max(1, self.stage_canvas.winfo_height())
        self.stage_canvas.delete("all")

        if self.subject_img is None:
            self.stage_canvas.create_text(
                cw//2, ch//2,
                text="Upload a subject photo to preview",
                fill=TEXT_DIM,
                font=(self.font_family, 14, "bold")
            )
            return

        base = self.subject_img.copy().convert("RGBA")
        live = composite_tryon(base, self.garment_img, fit=self.fit, x_off=self.x_off, y_off=0, opacity=self.opacity) if self.garment_img else base
        show = self.result_img if (final and self.result_img is not None) else live

        # ✅ Fill stage: crop to stage aspect before resizing
        show = center_crop_to_aspect(show, cw - 20, ch - 20)

        w, h = show.size
        scale = min((cw - 20) / w, (ch - 20) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        disp = show.resize((nw, nh), Image.LANCZOS)

        self.stage_tk = ImageTk.PhotoImage(disp)
        self.stage_canvas.create_image(cw//2, ch//2, image=self.stage_tk)

    def _render_debug(self):
        if self.subject_img is None:
            for tile in self.debug_tiles:
                tile.canvas.delete("all")
                tile.canvas.create_text(42, 42, text="—", fill=TEXT_DIM)
            return

        orig = self.subject_img.copy().convert("RGBA")

        if self.backend_debug is not None:
            overlay_rgb, mask_rgb, _info = self.backend_debug

            if mask_rgb is not None:
                mask_rgba = mask_rgb.convert("RGBA")
            else:
                mask_rgba = Image.new("RGBA", orig.size, (0, 0, 0, 255))

            if overlay_rgb is not None:
                mesh = overlay_rgb.convert("RGBA")
            else:
                live = self.result_img.convert("RGBA") if self.result_img is not None else orig
                mesh = mesh_overlay(live, color=ACCENT_CYAN)

            imgs = [orig, mask_rgba, mesh]
        else:
            if self.garment_img is not None:
                mask_rgba = garment_mask_preview(self.garment_img)
            else:
                mask_rgba = Image.new("RGBA", orig.size, (0, 0, 0, 255))

            live = composite_tryon(orig, self.garment_img, fit=self.fit, x_off=self.x_off, y_off=0, opacity=self.opacity) if self.garment_img else orig
            mesh = mesh_overlay(live, color=ACCENT_CYAN)
            imgs = [orig, mask_rgba, mesh]

        for tile, im in zip(self.debug_tiles, imgs):
            tile.canvas.delete("all")
            w, h = im.size
            scale = min(84 / w, 84 / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            disp = im.resize((nw, nh), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(disp)
            tile.canvas.image = tkimg
            tile.canvas.create_image(42, 42, image=tkimg)


if __name__ == "__main__":
    app = TryOnStudio()
    app.mainloop()
