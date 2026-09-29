# app.py

import streamlit as st
import os
import numpy as np
from PIL import Image

from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
import torch
import ollama

from src.cv_pipeline import analyze_wound  # <- now uses Deepskin under the hood

# ── Paths ───────────────────────────────────────────────────────
QDRANT_PATH = r"C:\Users\manoj\Desktop\WoundTrack-RAG\data\qdrant_db"
COLLECTION = "wound_knowledge"
OUTPUT_DIR = r"C:\Users\manoj\Desktop\WoundTrack-RAG\data\images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="WoundTrack-RAG", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .report-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;
                  border-left: 4px solid #1f77b4; font-size: 0.95rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... please wait"):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        client = QdrantClient(path=QDRANT_PATH)
    return model, processor, device, client


def get_text_embedding(text, model, processor, device):
    inputs = processor(text=[text[:77]], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        features = model.text_model(**inputs).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


def search_medical_knowledge(query, model, processor, device, client, top_k=3):
    query_vec = get_text_embedding(query, model, processor, device)
    results = client.query_points(collection_name=COLLECTION, query=query_vec, limit=top_k).points
    return [{"text": r.payload["text"], "source": r.payload["source"],
             "page": r.payload["page"], "score": round(r.score, 3)} for r in results]


def generate_report(cv_results, retrieved_docs, filename):
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"\n[Source {i+1}: {doc['source']}, Page {doc['page']}]\n{doc['text']}\n"

    pwat_line = f"- PWAT Score: {cv_results['pwat_score']:.2f}" if cv_results.get("pwat_score") is not None else ""

    prompt = f"""You are a clinical wound care specialist AI assistant.

WOUND ANALYSIS RESULTS:
- Wound Area: {cv_results['wound_area_pixels']:.0f} pixels
- Tissue Type: {cv_results['tissue_type']}
{pwat_line}
- Image: {filename}

RETRIEVED MEDICAL KNOWLEDGE:
{context}

Generate a structured clinical report with:
1. WOUND ASSESSMENT
2. TISSUE ANALYSIS
3. TREATMENT RECOMMENDATIONS (evidence-based)
4. CITATIONS

Keep it professional and concise."""

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def get_tissue_color(tissue):
    return {"granulation": "🟢", "slough": "🟡", "necrotic": "🔴", "unknown": "⚪"}.get(tissue, "⚪")


def main():
    st.markdown('<div class="main-title">🩺 WoundTrack-RAG</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:gray;">Multimodal Wound Progress Intelligence — Pretrained U-Net Segmentation + RAG</div>', unsafe_allow_html=True)

    model, processor, device, client = load_models()
    st.success(f"✅ AI Models loaded on {device.upper()}")

    with st.sidebar:
        st.title("WoundTrack-RAG")
        st.markdown("### Model Info:")
        st.markdown("- **Segmentation:** Deepskin (pretrained U-Net)")
        st.markdown("- **Retrieval:** CLIP ViT-L/14 + Qdrant")
        st.markdown("- **LLM:** Llama 3.2 (Ollama)")

    st.markdown("### Upload Wound Image")
    uploaded_file = st.file_uploader("Choose a wound image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        save_dir = "data/images"
        os.makedirs(save_dir, exist_ok=True)
        input_path = os.path.join(save_dir, "uploaded_wound.jpg")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original Image")
            st.image(Image.open(input_path), use_container_width=True)

        if st.button("🔬 Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Running Deepskin segmentation..."):
                cv_results = analyze_wound(input_path, output_path="data/images/segmented_output.jpg")

            with col2:
                st.markdown("#### Segmented Image (Deepskin U-Net)")
                if os.path.exists(cv_results["segmented_image_path"]):
                    st.image(Image.open(cv_results["segmented_image_path"]), use_container_width=True)

            st.markdown("---")
            st.markdown("### 📊 Wound Analysis Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Wound Area", f"{cv_results['wound_area_pixels']:.0f} px")
            tissue = cv_results["tissue_type"]
            m2.metric("Tissue Type", f"{get_tissue_color(tissue)} {tissue.capitalize()}")
            pwat = cv_results.get("pwat_score")
            m3.metric("PWAT Score", f"{pwat:.2f}" if pwat is not None else "N/A",
                       help="Photographic Wound Assessment Tool — predicted clinical severity score")

            st.markdown("---")
            st.markdown("### 📚 Retrieved Medical Knowledge")
            with st.spinner("Searching medical knowledge base..."):
                query = f"{tissue} wound treatment recommendations"
                retrieved = search_medical_knowledge(query, model, processor, device, client)

            for i, doc in enumerate(retrieved):
                with st.expander(f"📄 Source {i+1}: {doc['source']} (Page {doc['page']}) — Score: {doc['score']}"):
                    st.write(doc["text"])

            st.markdown("---")
            st.markdown("### 📋 AI Clinical Report")
            with st.spinner("Generating clinical report with Ollama..."):
                report = generate_report(cv_results, retrieved, uploaded_file.name)

            st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

            st.download_button("⬇️ Download Clinical Report", data=report,
                                file_name="wound_clinical_report.txt", mime="text/plain")


if __name__ == "__main__":
    main()