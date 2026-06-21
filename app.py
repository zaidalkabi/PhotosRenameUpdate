import streamlit as st
import easyocr
import cv2
import numpy as np
import zipfile
import os
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Page Configuration
st.set_page_config(
    page_title="Smart Image Renamer Pro",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark/Modern UI
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .preview-card {
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache OCR Reader to prevent reloading
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], model_storage_directory='/tmp', gpu=False)

reader = load_ocr_reader()

# Worker function for Fast Parallel Processing
def process_single_image(file, pattern_type):
    # Read file bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Speed Optimization: Resize a temporary copy for OCR if it's too large
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_for_ocr = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_for_ocr = img

    # Run OCR
    results = reader.readtext(img_for_ocr)
    
    detected_name = None
    for (bbox, text, prob) in results:
        clean_text = text.strip()
        
        if pattern_type == "Starts with CA (e.g., ca5_3)":
            if re.search(r'(?i)ca[\d_\-]+', clean_text):
                detected_name = clean_text
                break
        elif pattern_type == "Any text containing numbers":
            if re.search(r'[A-Za-z]+.*\d+|\d+.*[A-Za-z]+', clean_text):
                detected_name = clean_text
                break
        else:
            if len(clean_text) > 2:
                detected_name = clean_text
                break
                
    status = "✅ Text detected successfully!" if detected_name else "⚠️ No text matched - Kept original name"
    
    if not detected_name:
        detected_name = os.path.splitext(file.name)[0]
        
    # Clean file name from illegal characters
    detected_name = re.sub(r'[\\/*?:"<>| ]', "_", detected_name)
    ext = os.path.splitext(file.name)[1].lower()
    
    return {
        "original_name": file.name,
        "suggested_name": detected_name,
        "extension": ext,
        "image_data": img_rgb,
        "bytes": file_bytes,
        "status": status
    }

# --- Sidebar Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1086/1086741.png", width=80)
    st.title("⚙️ Control Panel")
    st.write("Optimize your extraction rules:")
    
    pattern_type = st.selectbox(
        "Target Text Pattern:",
        ["Starts with CA (e.g., ca5_3)", "Any text containing numbers", "Extract any text found"]
    )
    
    st.markdown("---")
    st.success("⚡ **Turbo Mode Active:** Images are processed in parallel for maximum speed.")

# --- Main Interface ---
st.title("🏷️ Smart Image Renamer Pro")
st.caption("Upload your project/site images. AI will instantly scan codes, rename, and pack them into a ZIP file.")

uploaded_files = st.file_uploader(
    "Drag and drop your images here (Supports JPG, PNG)", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.subheader(f"📸 Uploaded Files ({len(uploaded_files)})")
    
    processed_images = []
    
    # Progress feedback
    with st.spinner("⚡ Running high-speed parallel OCR detection..."):
        # Executing multi-threaded extraction
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_image, file, pattern_type) for file in uploaded_files]
            processed_images = [future.result() for future in futures]
            
    st.toast("🎉 Fast scan completed!", icon="🚀")
    
    # --- Interactive Preview & Editing Grid ---
    st.markdown("### 👁️ Review & Edit Suggested Names")
    st.write("You can instantly overwrite or tweak any name in the input boxes below before final download.")
    
    final_files_to_zip = {}
    
    # Display in a clean 2-column grid layout
    cols = st.columns(2)
    for i, item in enumerate(processed_images):
        col = cols[i % 2]
        with col:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            
            sub_col1, sub_col2 = st.columns([1, 2])
            with sub_col1:
                st.image(item["image_data"], use_container_width=True)
            with sub_col2:
                st.markdown(f"**Original:** `{item['original_name']}`")
                st.markdown(f"<small>{item['status']}</small>", unsafe_allow_html=True)
                
                user_edited_name = st.text_input(
                    f"Final Name for Image #{i+1}", 
                    value=item["suggested_name"],
                    key=f"input_{i}"
                )
                
                # Prevent duplicate file names inside ZIP
                final_name = f"{user_edited_name}{item['extension']}"
                if final_name in final_files_to_zip:
                    final_name = f"{user_edited_name}_{i}{item['extension']}"
                    
                final_files_to_zip[final_name] = item["bytes"]
                
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Final Output ---
    st.markdown("---")
    st.subheader("📦 Build & Download Package")
    
    if st.button("🚀 Generate Renamed ZIP File"):
        zip_path = "renamed_images_package.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename, file_bytes in final_files_to_zip.items():
                zipf.writestr(filename, file_bytes)
                
        st.balloons()
        
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📥 Click Here to Download Your ZIP File",
                data=f,
                file_name="renamed_site_images.zip",
                mime="application/zip",
                use_container_width=True
            )
