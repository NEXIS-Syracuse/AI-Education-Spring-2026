"""
NEXIS Education — PDF Audio Reader
Streamlit app that converts PDF pages to narrated audio using Kokoro TTS.
Handles text extraction, image captioning (BLIP), OCR (EasyOCR), and TTS (Kokoro).
"""

import streamlit as st
import os
import io
import re
import time
import tempfile
import numpy as np
import soundfile as sf
import torch
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image
from kokoro import KPipeline
# COMMENTED OUT at top-level — lazy-loaded inside load_blip() to avoid
# massive __path__ warning spam and ~100MB RAM overhead at startup.
# from transformers import (
#     BlipProcessor,
#     BlipForConditionalGeneration,
#     # pipeline as hf_pipeline,  # only BART used this (~1.6GB RAM)
# )
# import easyocr  # lazy-loaded inside load_ocr()

# **SPEED: detect GPU once at startup**
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXIS Education — PDF Audio Reader",
    page_icon="🎧",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
# **SPEED: cap text sent to TTS to avoid synthesising huge pages**
MAX_TTS_CHARS = 3000
# **SPEED: resize images to this max dimension before captioning/OCR**
MAX_IMG_DIM = 512
# ── COMMENTED OUT: BART classifier labels & prompts (saves ~1.6GB RAM) ──────
# To re-enable, uncomment these and the load_classifier / classify_image functions.
# IMAGE_LABELS = [
#     "graph or chart",
#     "flowchart or diagram",
#     "infographic",
#     "user interface screenshot",
#     "photograph or illustration",
#     "table",
#     "equation or formula",
# ]
# ROUTE_PROMPTS = {
#     "graph or chart": (
#         "You are narrating a graph or chart for a blind listener. "
#         "Describe the trends, axes, and key data points clearly.\n"
#         "Caption: {caption}\nText in image: {ocr}\nDescription:"
#     ),
#     "flowchart or diagram": (
#         "You are narrating a flowchart or diagram for a blind listener. "
#         "Walk through the steps or components in order.\n"
#         "Caption: {caption}\nText in image: {ocr}\nDescription:"
#     ),
#     "infographic": (
#         "You are narrating an infographic for a blind listener. "
#         "Summarise the key facts and their relationships.\n"
#         "Caption: {caption}\nText in image: {ocr}\nDescription:"
#     ),
#     "user interface screenshot": (
#         "You are narrating a UI screenshot for a blind listener. "
#         "Describe the layout, buttons, and content visible.\n"
#         "Caption: {caption}\nText in image: {ocr}\nDescription:"
#     ),
#     "table": (
#         "You are narrating a table for a blind listener. "
#         "Read out the column headers and the most important rows.\n"
#         "Caption: {caption}\nText in image: {ocr}\nDescription:"
#     ),
#     "equation or formula": (
#         "Read this equation aloud clearly.\n"
#         "Caption: {caption}\nText in image: {ocr}\nEquation spoken:"
#     ),
# }
# DEFAULT_PROMPT = (
#     "Describe this image for a blind listener.\n"
#     "Caption: {caption}\nText in image: {ocr}\nDescription:"
# )

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so they load once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TTS model (Kokoro)…")
def load_tts():
    return KPipeline(lang_code="a")


# ── BLIP and EasyOCR: lazy-loaded only when images are found on a page ──────
# This keeps RAM at ~700MB for text-only PDFs (Kokoro + PyTorch only).
@st.cache_resource(show_spinner="Loading image captioning model (BLIP)…")
def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration  # lazy import
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    # **SPEED: move BLIP to GPU if available for faster inference**
    ).to(DEVICE)
    return processor, model


@st.cache_resource(show_spinner="Loading OCR engine (EasyOCR)…")
def load_ocr():
    import easyocr  # lazy import to avoid loading at startup
    # **SPEED: gpu=True uses CUDA if available; falls back to CPU automatically**
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


# ── COMMENTED OUT: BART zero-shot classifier (~1.6GB RAM — too heavy for Cloud) ──
# @st.cache_resource(show_spinner="Loading zero-shot image classifier…")
# def load_classifier():
#     device = 0 if torch.cuda.is_available() else -1
#     return hf_pipeline(
#         "zero-shot-classification",
#         model="facebook/bart-large-mnli",
#         device=device,
#     )


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def blip_caption(pil_img: Image.Image, processor, model) -> str:
    # **SPEED: no_grad skips gradient tracking, cutting memory and time**
    with torch.no_grad():
        inputs = processor(pil_img, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(out[0], skip_special_tokens=True)


def resize_image(pil_img: Image.Image, max_dim: int = MAX_IMG_DIM) -> Image.Image:
    """**SPEED: downsample large images before captioning/OCR — major time saver.**"""
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


def ocr_text(pil_img: Image.Image, reader) -> str:
    # **SPEED: detail=0 skips bounding-box computation, noticeably faster**
    result = reader.readtext(np.array(pil_img), detail=0)
    return " ".join(result).strip()


# ── COMMENTED OUT: classify_image (requires BART — too heavy for Cloud) ──────
# def classify_image(caption: str, ocr: str, classifier) -> str:
#     probe = f"{caption}. {ocr[:300]}"
#     result = classifier(probe, IMAGE_LABELS, multi_label=False)
#     return result["labels"][0]


def describe_image(pil_img: Image.Image, blip_proc, blip_model, ocr_reader) -> tuple[str, str]:
    """Describe an image using BLIP captioning + OCR (no BART classification)."""
    # **SPEED: resize before any inference**
    pil_img = resize_image(pil_img)
    caption = blip_caption(pil_img, blip_proc, blip_model)
    ocr = ocr_text(pil_img, ocr_reader)
    # ── COMMENTED OUT: BART classification ──
    # label = classify_image(caption, ocr, classifier)
    # template = ROUTE_PROMPTS.get(label, DEFAULT_PROMPT)
    # description = f"{label}. {caption}."
    # Without BART, we just use the BLIP caption directly:
    description = caption
    if ocr:
        description += f". Text in image: {ocr[:400]}"
    return "image", description


def extract_images_from_page(fitz_page, fitz_doc) -> list[Image.Image]:
    imgs = []
    for img_info in fitz_page.get_images(full=True):
        xref = img_info[0]
        base = fitz_doc.extract_image(xref)
        pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
        if pil.width >= 80 and pil.height >= 80:
            imgs.append(pil)
    return imgs


def tts_synthesize(text: str, kokoro_pipeline, voice: str) -> np.ndarray:
    # **SPEED: truncate at sentence boundary near MAX_TTS_CHARS to keep TTS fast**
    if len(text) > MAX_TTS_CHARS:
        cutoff = text.rfind(".", 0, MAX_TTS_CHARS)
        text = text[: cutoff + 1] if cutoff != -1 else text[:MAX_TTS_CHARS]
    chunks = []
    for _, _, audio in kokoro_pipeline(text, voice=voice):
        chunks.append(audio)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def audio_to_bytes(audio: np.ndarray) -> bytes:
    # **AUDIO FIX: write to a fresh BytesIO and explicitly seek(0) before returning**
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def mini_audio_player(audio_bytes: bytes, page_num: int) -> None:
    """Compact mini audio player with play/pause, seek, speed, and volume."""
    import base64
    b64 = base64.b64encode(audio_bytes).decode()

    st.components.v1.html(f"""
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{ background: transparent; }}
      .mp {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 10px 14px;
        color: #e0e0e0;
      }}
      .mp-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
      .mp-lbl {{ font-size: 11px; color: #8899aa; flex: 1; }}
      .mp-spd {{
        font-size: 10px; font-weight: 600; color: #64ffda;
        background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.25);
        border-radius: 4px; padding: 2px 6px; cursor: pointer;
        transition: background 0.2s;
      }}
      .mp-spd:hover {{ background: rgba(100,255,218,0.2); }}
      .mp-row {{ display: flex; align-items: center; gap: 8px; }}
      .mp-btn {{
        width: 32px; height: 32px; border-radius: 50%;
        border: none; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        background: rgba(100,255,218,0.15); transition: background 0.2s;
        flex-shrink: 0;
      }}
      .mp-btn:hover {{ background: rgba(100,255,218,0.3); }}
      .mp-btn svg {{ fill: #64ffda; }}
      .mp-mid {{ flex: 1; display: flex; flex-direction: column; gap: 2px; }}
      .mp-bar {{
        -webkit-appearance: none; appearance: none;
        width: 100%; height: 4px; border-radius: 2px;
        background: rgba(255,255,255,0.12); outline: none; cursor: pointer;
      }}
      .mp-bar::-webkit-slider-thumb {{
        -webkit-appearance: none; width: 12px; height: 12px;
        border-radius: 50%; background: #64ffda; border: none; cursor: pointer;
      }}
      .mp-times {{ display: flex; justify-content: space-between; font-size: 10px; color: #667788; }}
      .mp-vol {{ display: flex; align-items: center; gap: 4px; flex-shrink: 0; }}
      .mp-vol svg {{ fill: #667788; }}
      .mp-vbar {{
        -webkit-appearance: none; appearance: none;
        width: 40px; height: 3px; border-radius: 2px;
        background: rgba(255,255,255,0.12); outline: none; cursor: pointer;
      }}
      .mp-vbar::-webkit-slider-thumb {{
        -webkit-appearance: none; width: 10px; height: 10px;
        border-radius: 50%; background: #667788; border: none; cursor: pointer;
      }}
    </style>

    <div class="mp">
      <div class="mp-top">
        <span class="mp-lbl" id="lbl{page_num}">Page {page_num}</span>
        <button class="mp-spd" id="spd{page_num}" onclick="cycleSpeed()">1×</button>
      </div>
      <div class="mp-row">
        <button class="mp-btn" id="playbtn{page_num}" onclick="togglePlay()">
          <svg width="14" height="14" viewBox="0 0 24 24" id="icon{page_num}">
            <path d="M8 5v14l11-7z"/>
          </svg>
        </button>
        <div class="mp-mid">
          <input type="range" class="mp-bar" id="prog{page_num}" min="0" max="100" step="0.1" value="0"
            oninput="onScrub(this.value)">
          <div class="mp-times">
            <span id="cur{page_num}">0:00</span>
            <span id="dur{page_num}">--:--</span>
          </div>
        </div>
        <div class="mp-vol">
          <svg width="12" height="12" viewBox="0 0 24 24">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
          </svg>
          <input type="range" class="mp-vbar" id="vol{page_num}" min="0" max="1" step="0.01" value="0.8"
            oninput="onVol(this.value)">
        </div>
      </div>
      <audio id="aud{page_num}" style="display:none"></audio>
    </div>

    <script>
      const SPEEDS = [0.75, 1, 1.25, 1.5, 2];
      let speedIdx = 1;
      const aud = document.getElementById('aud{page_num}');
      const prog = document.getElementById('prog{page_num}');
      const icon = document.getElementById('icon{page_num}');
      const lbl  = document.getElementById('lbl{page_num}');
      const curEl = document.getElementById('cur{page_num}');
      const durEl = document.getElementById('dur{page_num}');

      const b64 = "{b64}";
      const blob = new Blob(
        [Uint8Array.from(atob(b64), c => c.charCodeAt(0))],
        {{type: 'audio/wav'}}
      );
      aud.src = URL.createObjectURL(blob);

      function fmt(s) {{
        s = Math.round(s);
        return Math.floor(s/60) + ':' + String(s%60).padStart(2,'0');
      }}

      aud.addEventListener('loadedmetadata', () => {{
        prog.max = aud.duration;
        durEl.textContent = fmt(aud.duration);
        lbl.textContent = 'Page {page_num} — ' + fmt(aud.duration);
      }});

      aud.addEventListener('timeupdate', () => {{
        prog.value = aud.currentTime;
        curEl.textContent = fmt(aud.currentTime);
      }});

      aud.addEventListener('ended', () => {{
        icon.innerHTML = '<path d="M8 5v14l11-7z"/>';
        prog.value = 0;
        curEl.textContent = '0:00';
      }});

      function togglePlay() {{
        if (aud.paused) {{
          aud.play();
          icon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
        }} else {{
          aud.pause();
          icon.innerHTML = '<path d="M8 5v14l11-7z"/>';
        }}
      }}

      function doSkip(secs) {{
        aud.currentTime = Math.min(Math.max(0, aud.currentTime + secs), aud.duration);
      }}

      function onScrub(v) {{
        aud.currentTime = v;
        curEl.textContent = fmt(v);
      }}

      function onVol(v) {{ aud.volume = v; }}

      function cycleSpeed() {{
        speedIdx = (speedIdx + 1) % SPEEDS.length;
        aud.playbackRate = SPEEDS[speedIdx];
        document.getElementById('spd{page_num}').textContent = SPEEDS[speedIdx] + '×';
      }}
    </script>
    """, height=90)


def _log_step(container, log_lines: list[str], msg: str, elapsed: float | None = None):
    """Append a timestamped line to the progress log and re-render it."""
    suffix = f"  `({elapsed:.1f}s)`" if elapsed is not None else " ⏳"
    log_lines.append(f"- {msg}{suffix}")
    container.markdown("\n".join(log_lines))


def process_page(
    page_idx: int,
    pdf_text: PdfReader,
    pdf_fitz,
    kokoro_pipeline,
    voice: str,
    log_container,
    log_lines: list[str],
) -> tuple[str, np.ndarray]:
    """Process a single PDF page → (narration text, audio array)."""

    # 1. Text extraction
    t0 = time.time()
    _log_step(log_container, log_lines, f"📝 **Page {page_idx + 1}** — extracting text")
    page_text = pdf_text.pages[page_idx].extract_text() or ""
    page_text = re.sub(r"\n+", " ", page_text).strip()
    _log_step(log_container, log_lines,
              f"  Text: {len(page_text)} chars", time.time() - t0)

    # 2. Images — BLIP & EasyOCR are lazy-loaded ONLY if images exist
    t0 = time.time()
    fitz_page = pdf_fitz[page_idx]
    page_images = extract_images_from_page(fitz_page, pdf_fitz)
    _log_step(log_container, log_lines,
              f"  Found **{len(page_images)}** image(s)", time.time() - t0)

    image_narrations = []
    if page_images:
        # Lazy-load BLIP and EasyOCR only when we actually need them
        _log_step(log_container, log_lines, "  ⬇️ Loading image models (BLIP + OCR)…")
        t0 = time.time()
        blip_proc, blip_model = load_blip()
        ocr_reader = load_ocr()
        _log_step(log_container, log_lines,
                  "  Image models ready", time.time() - t0)

        for img_i, pil in enumerate(page_images):
            t0 = time.time()
            _log_step(log_container, log_lines,
                      f"  🖼️ Describing image {img_i + 1}/{len(page_images)}")
            label, desc = describe_image(pil, blip_proc, blip_model, ocr_reader)
            spoken = f"Image {img_i + 1}: {desc}"
            image_narrations.append(spoken)
            _log_step(log_container, log_lines,
                      f"  Image {img_i + 1} described", time.time() - t0)

    # 3. Combine
    parts = []
    if page_text:
        parts.append(page_text)
    parts.extend(image_narrations)

    if not parts:
        _log_step(log_container, log_lines,
                  f"  ⚠️ Page {page_idx + 1}: no extractable content", 0.0)
        return "(no content)", np.zeros(0, dtype=np.float32)

    full_narration = "  ".join(parts)

    # 4. TTS
    t0 = time.time()
    _log_step(log_container, log_lines,
              f"  🔊 Synthesising audio ({len(full_narration)} chars)")
    audio = tts_synthesize(full_narration, kokoro_pipeline, voice)
    _log_step(log_container, log_lines,
              f"  ✅ Audio ready — {len(audio)/SAMPLE_RATE:.1f}s of audio",
              time.time() - t0)

    return full_narration, audio


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎧 NEXIS Education — PDF Audio Reader")
st.markdown(
    "Upload a PDF and convert its pages into narrated audio. "
    "Images are automatically captioned and described for accessibility."
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    voice = st.selectbox(
        "TTS Voice",
        options=[
            "af_heart", "af_alloy", "af_echo", "af_fable",
            "af_nova", "af_onyx", "af_shimmer", "am_adam", "am_echo",
        ],
        index=0,
        help="Kokoro voice to use for narration",
    )
    page_range_mode = st.radio(
        "Pages to process",
        options=["All pages", "Page range", "Single page"],
        index=0,
    )
    start_page = end_page = single_page = None
    if page_range_mode == "Page range":
        col1, col2 = st.columns(2)
        start_page = col1.number_input("From page", min_value=1, value=1, step=1)
        end_page = col2.number_input("To page", min_value=1, value=1, step=1)
    elif page_range_mode == "Single page":
        single_page = st.number_input("Page number", min_value=1, value=1, step=1)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Uses [Kokoro TTS](https://github.com/hexgrad/kokoro), "
        "BLIP image captioning, EasyOCR, and PyMuPDF."
    )

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save to temp file so fitz and pypdf can both open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    pdf_text = PdfReader(tmp_path)
    pdf_fitz = fitz.open(tmp_path)
    num_pages = len(pdf_text.pages)

    st.success(f"Loaded **{uploaded_file.name}** — {num_pages} page(s)")

    # Determine which pages to process
    if page_range_mode == "All pages":
        pages_to_process = list(range(num_pages))
    elif page_range_mode == "Page range":
        s = max(1, int(start_page)) - 1
        e = min(num_pages, int(end_page))
        pages_to_process = list(range(s, e))
    else:  # Single page
        p = min(num_pages, max(1, int(single_page))) - 1
        pages_to_process = [p]

    st.info(
        f"Will process {len(pages_to_process)} page(s): "
        + ", ".join(str(p + 1) for p in pages_to_process)
    )

    if st.button("🚀 Generate Audio", type="primary"):
        overall_start = time.time()

        # ── Model loading ─────────────────────────────────────────────────
        # Only Kokoro TTS loads eagerly (~400MB).
        # BLIP + EasyOCR lazy-load inside process_page IF images are found.
        # BART classifier is commented out entirely (saves ~1.6GB).
        progress = st.progress(0.0, text="Loading TTS model (Kokoro)…")
        log_container = st.container()
        log_lines: list[str] = []
        _log_step(log_container, log_lines, "🔧 **Loading TTS model** (first run downloads & caches it)")

        t0 = time.time()
        kokoro_pipeline = load_tts()
        _log_step(log_container, log_lines,
                  "🗣️ Kokoro TTS — ready", time.time() - t0)

        # ── COMMENTED OUT: eager loading of BLIP, EasyOCR, BART ──────────
        # These are now lazy-loaded in process_page only when images exist.
        # blip_proc, blip_model = load_blip()
        # ocr_reader = load_ocr()
        # classifier = load_classifier()  # BART — removed entirely

        progress.progress(0.3, text="TTS model loaded. Processing pages…")
        _log_step(log_container, log_lines, "---")
        _log_step(log_container, log_lines, "📄 **Processing pages**")

        # ── Page processing ──────────────────────────────────────────────
        st.session_state["results"] = {}
        n = len(pages_to_process)

        for i, page_idx in enumerate(pages_to_process):
            page_start = time.time()
            frac = 0.3 + 0.7 * (i / n)
            progress.progress(
                frac,
                text=f"Processing page {page_idx + 1} ({i + 1}/{n})…",
            )

            narration, audio = process_page(
                page_idx=page_idx,
                pdf_text=pdf_text,
                pdf_fitz=pdf_fitz,
                kokoro_pipeline=kokoro_pipeline,
                voice=voice,
                log_container=log_container,
                log_lines=log_lines,
            )
            st.session_state["results"][page_idx] = {
                "narration": narration,
                "audio_bytes": audio_to_bytes(audio),
            }

        total = time.time() - overall_start
        progress.progress(1.0, text=f"Done! Total time: {total:.1f}s")
        _log_step(log_container, log_lines,
                  f"🎉 **All done** — {n} page(s) in {total:.1f}s", total)

    # Display results (outside button block so they persist across reruns)
    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]
        st.success(f"✅ Processed {len(results)} page(s)!")
        st.markdown("---")
        st.header("📄 Results")

        for page_idx in sorted(results.keys()):
            data = results[page_idx]
            with st.expander(f"Page {page_idx + 1}", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Narration text:**")
                    st.text_area(
                        label="",
                        value=data["narration"],
                        height=150,
                        key=f"text_{page_idx}",
                        label_visibility="collapsed",
                    )
                with col2:
                    st.markdown("**Audio:**")
                    # Use mini player if audio has content, otherwise show info
                    if len(data["audio_bytes"]) <= 44:
                        st.info("No audio — page had no extractable content.")
                    else:
                        mini_audio_player(data["audio_bytes"], page_idx + 1)
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=data["audio_bytes"],
                        file_name=f"page_{page_idx + 1:03}.wav",
                        mime="audio/wav",
                        key=f"dl_{page_idx}",
                    )

        # Clean up temp file (may already be gone on rerun)
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

else:
    st.info("👆 Upload a PDF to get started.")
    st.markdown(
        """
        ### What this app does
        1. **Extracts text** from each PDF page
        2. **Detects and describes images** using BLIP captioning + EasyOCR (loaded only when needed)
        3. **Synthesises audio** with Kokoro TTS
        4. **Plays and exports** per-page WAV files
        """
    )