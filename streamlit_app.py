"""
NEXIS Education — PDF Audio Reader
Streamlit app that converts PDF pages to narrated audio using Kokoro TTS.
Handles text extraction, image captioning (BLIP2), OCR (EasyOCR), and TTS (Kokoro).
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
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline as hf_pipeline,
)
import easyocr

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
IMAGE_LABELS = [
    "graph or chart",
    "flowchart or diagram",
    "infographic",
    "user interface screenshot",
    "photograph or illustration",
    "table",
    "equation or formula",
]
ROUTE_PROMPTS = {
    "graph or chart": (
        "You are narrating a graph or chart for a blind listener. "
        "Describe the trends, axes, and key data points clearly.\n"
        "Caption: {caption}\nText in image: {ocr}\nDescription:"
    ),
    "flowchart or diagram": (
        "You are narrating a flowchart or diagram for a blind listener. "
        "Walk through the steps or components in order.\n"
        "Caption: {caption}\nText in image: {ocr}\nDescription:"
    ),
    "infographic": (
        "You are narrating an infographic for a blind listener. "
        "Summarise the key facts and their relationships.\n"
        "Caption: {caption}\nText in image: {ocr}\nDescription:"
    ),
    "user interface screenshot": (
        "You are narrating a UI screenshot for a blind listener. "
        "Describe the layout, buttons, and content visible.\n"
        "Caption: {caption}\nText in image: {ocr}\nDescription:"
    ),
    "table": (
        "You are narrating a table for a blind listener. "
        "Read out the column headers and the most important rows.\n"
        "Caption: {caption}\nText in image: {ocr}\nDescription:"
    ),
    "equation or formula": (
        "Read this equation aloud clearly.\n"
        "Caption: {caption}\nText in image: {ocr}\nEquation spoken:"
    ),
}
DEFAULT_PROMPT = (
    "Describe this image for a blind listener.\n"
    "Caption: {caption}\nText in image: {ocr}\nDescription:"
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so they load once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TTS model (Kokoro)…")
def load_tts():
    return KPipeline(lang_code="a")


@st.cache_resource(show_spinner="Loading image captioning model (BLIP)…")
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


@st.cache_resource(show_spinner="Loading OCR engine (EasyOCR)…")
def load_ocr():
    return easyocr.Reader(["en"])


@st.cache_resource(show_spinner="Loading zero-shot image classifier…")
def load_classifier():
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def blip_caption(pil_img: Image.Image, processor, model) -> str:
    inputs = processor(pil_img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(out[0], skip_special_tokens=True)


def ocr_text(pil_img: Image.Image, reader) -> str:
    result = reader.readtext(np.array(pil_img))
    return " ".join([item[1] for item in result]).strip()


def classify_image(caption: str, ocr: str, classifier) -> str:
    probe = f"{caption}. {ocr[:300]}"
    result = classifier(probe, IMAGE_LABELS, multi_label=False)
    return result["labels"][0]


def describe_image(pil_img: Image.Image, blip_proc, blip_model, ocr_reader, classifier) -> tuple[str, str]:
    caption = blip_caption(pil_img, blip_proc, blip_model)
    ocr = ocr_text(pil_img, ocr_reader)
    label = classify_image(caption, ocr, classifier)
    template = ROUTE_PROMPTS.get(label, DEFAULT_PROMPT)
    # For Streamlit we use the caption + OCR directly as the narration
    # (avoids loading a heavy T5 model; BLIP caption is already descriptive)
    description = f"{label}. {caption}."
    if ocr:
        description += f" Text in image: {ocr[:400]}"
    return label, description


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
    chunks = []
    for _, _, audio in kokoro_pipeline(text, voice=voice):
        chunks.append(audio)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def audio_to_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def process_page(
    page_idx: int,
    pdf_text: PdfReader,
    pdf_fitz,
    blip_proc,
    blip_model,
    ocr_reader,
    classifier,
    kokoro_pipeline,
    voice: str,
    status_area,
) -> tuple[str, np.ndarray]:
    """Process a single PDF page → (narration text, audio array)."""

    # 1. Text
    page_text = pdf_text.pages[page_idx].extract_text() or ""
    page_text = re.sub(r"\n+", " ", page_text).strip()

    # 2. Images
    fitz_page = pdf_fitz[page_idx]
    page_images = extract_images_from_page(fitz_page, pdf_fitz)
    status_area.write(f"  Found **{len(page_images)}** image(s) on this page.")

    image_narrations = []
    for img_i, pil in enumerate(page_images):
        status_area.write(f"  → Describing image {img_i + 1}…")
        label, desc = describe_image(pil, blip_proc, blip_model, ocr_reader, classifier)
        spoken = f"Image {img_i + 1}, {label}. {desc}"
        image_narrations.append(spoken)

    # 3. Combine
    parts = []
    if page_text:
        parts.append(page_text)
    parts.extend(image_narrations)

    if not parts:
        return "(no content)", np.zeros(0, dtype=np.float32)

    full_narration = "  ".join(parts)

    # 4. TTS
    status_area.write("  Synthesising audio…")
    audio = tts_synthesize(full_narration, kokoro_pipeline, voice)

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
    st.header("Settings")
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

    if st.button("Generate Audio", type="primary"):
        # Load models
        kokoro_pipeline = load_tts()
        blip_proc, blip_model = load_blip()
        ocr_reader = load_ocr()
        classifier = load_classifier()

        results = {}  # page_idx → {"narration": str, "audio_bytes": bytes}

        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        for i, page_idx in enumerate(pages_to_process):
            progress.progress(
                i / len(pages_to_process),
                text=f"Processing page {page_idx + 1} / {num_pages}…",
            )
            status.write(f"### Page {page_idx + 1}")

            narration, audio = process_page(
                page_idx=page_idx,
                pdf_text=pdf_text,
                pdf_fitz=pdf_fitz,
                blip_proc=blip_proc,
                blip_model=blip_model,
                ocr_reader=ocr_reader,
                classifier=classifier,
                kokoro_pipeline=kokoro_pipeline,
                voice=voice,
                status_area=status,
            )
            results[page_idx] = {
                "narration": narration,
                "audio_bytes": audio_to_bytes(audio),
            }

        progress.progress(1.0, text="Done!")
        status.empty()

        st.success(f"✅ Processed {len(results)} page(s)!")

        # Display results
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
                    st.audio(data["audio_bytes"], format="audio/wav")
                    st.download_button(
                        label=f"⬇️ Download WAV",
                        data=data["audio_bytes"],
                        file_name=f"page_{page_idx + 1:03}.wav",
                        mime="audio/wav",
                        key=f"dl_{page_idx}",
                    )

        # Clean up temp file
        os.unlink(tmp_path)

else:
    st.info("👆 Upload a PDF to get started.")
    st.markdown(
        """
        ### What this app does
        1. **Extracts text** from each PDF page
        2. **Detects and describes images** using BLIP captioning + EasyOCR
        3. **Classifies images** (chart, diagram, table, etc.) for appropriate narration
        4. **Synthesises audio** with Kokoro TTS
        5. **Plays and exports** per-page WAV files
        """
    )