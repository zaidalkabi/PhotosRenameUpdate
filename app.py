import streamlit as st
import easyocr
import cv2
import numpy as np
import zipfile
import os
import re

# Page Configuration
st.set_page_config(
    page_title="Smart Image Renamer Pro",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
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

# Safe Local Directory for OCR Models
MODEL_DIR = os.path.join(os.getcwd(), "ocr_models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Cache OCR Reader Safely
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], model_storage_directory=MODEL_DIR, gpu=False)

try:
    reader = load_ocr_reader()
except Exception as e:
    st.error(f"Failed to initialize OCR engine: {e}")

# Stable Image Processing function
def process_single_image(file, pattern_type):
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Optimize image size for fast OCR
        h, w = img.shape[:2]
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            img_for_ocr = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_for_ocr = img

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
    except Exception as e:
        return None

# --- Sidebar Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1086/1086741.png", width=80)
    st.title("⚙️ Control Panel")
    
    pattern_type = st.selectbox(
        "Target Text Pattern:",
        ["Starts with CA (e.g., ca5_3)", "Any text containing numbers", "Extract any text found"]
    )
    st.markdown("---")
    st.info("💡 Safe State Engine is active to prevent multi-file drop crashes.")

# --- Main Interface ---
st.title("🏷️ Smart Image Renamer Pro")
st.caption("Upload multiple project images. AI will instantly scan codes, rename, and pack them into a ZIP file.")

uploaded_files = st.file_uploader(
    "Drag and drop your images here (Supports JPG, PNG)", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png'],
    key="file_uploader"
)

# Initialize Session States to prevent reload crashes (Fixed line 138 here)
if "processed_data" not in st.session_state or st.session_state.get("last_uploaded_count") != len(uploaded_files or []):
    st.session_state["processed_data"] = []
    st.session_state["last_uploaded_count"] = len(uploaded_files or [])
    st.session_state["user_edits"] = {}

if uploaded_files:
    # Trigger processing only if the session state is empty (prevents re-running OCR on text input)
    if not st.session_state["processed_data"]:
        with st.spinner("⚡ Running high-speed OCR on all files..."):
            for file in uploaded_files:
                res = process_single_image(file, pattern_type)
                if res:
                    st.session_state["processed_data"].append(res)
            st.toast("🎉 Scan completed successfully!", icon="🚀")

    # --- Interactive Preview & Editing Grid ---
    st.markdown("### 👁️ Review & Edit Suggested Names")
    st.write("You can instantly overwrite or tweak any name in the input boxes below.")
    
    final_files_to_zip = {}
    
    cols = st.columns(2)
    for i, item in enumerate(st.session_state["processed_data"]):
        col = cols[i % 2]
        with col:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            
            sub_col1, sub_col2 = st.columns([1, 2])
            with sub_col1:
                st.image(item["image_data"], use_container_width=True)
            with sub_col2:
                st.markdown(f"**Original:** `{item['original_name']}`")
                st.markdown(f"<small>{item['status']}</small>", unsafe_allow_html=True)
                
                # Use session state to capture and lock input changes safely
                input_key = f"input_{item['original_name']}_{i}"
                user_edited_name = st.text_input(
                    f"Final Name for Image #{i+1}", 
                    value=st.session_state["user_edits"].get(input_key, item["suggested_name"]),
                    key=input_key
                )
                st.session_state["user_edits"][input_key] = user_edited_name
                
                # Prevent duplicates
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
        
        try:
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
        except Exception as zip_err:
            st.error(f"Error packaging files into ZIP: {zip_err}")
else:
    # Clear history if files are removed
    st.session_state["processed_data"] = []
    st.session_state["user_edits"] = {}
