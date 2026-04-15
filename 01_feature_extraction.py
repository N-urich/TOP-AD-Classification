# =================================================
# 01_feature_extraction.py - TOP Feature Extraction
# Three Orthogonal Planes (Axial/Coronal/Sagittal)
# =================================================
import numpy as np
import os
import nibabel as nib  # For loading real MRI (ADNI)

# ----------------------
# Define Paths (Match Your ModelWhale Setup)
# ----------------------
RAW_DATA_PATH = "/home/mw/AD_Thesis_FULL/1_TOP_Model/data/raw_mri"
PREPROCESSED_PATH = "/home/mw/AD_Thesis_FULL/1_TOP_Model/preprocessed"
FEATURE_SAVE_PATH = os.path.join(PREPROCESSED_PATH, "features")
ROI_SAVE_PATH = os.path.join(PREPROCESSED_PATH, "roi")

# Create folders if missing
os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)
os.makedirs(ROI_SAVE_PATH, exist_ok=True)

# ----------------------
# Core Preprocessing Functions
# ----------------------
def load_mri(file_path):
    """Load real NIfTI MRI (ADNI dataset) from GitHub repo"""
    img = nib.load(file_path)
    return img.get_fdata()

def skull_strip(vol, threshold=50):
    """Simulate BET skull stripping (remove non-brain tissue)"""
    mask = vol > threshold
    return vol * mask, mask

def normalize_intensity(vol):
    """Normalize MRI to [0, 1] (standard medical imaging step)"""
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

def extract_orthogonal_planes(vol):
    """Extract 3 orthogonal planes (Axial/Coronal/Sagittal)"""
    # Axial (middle slice of z-axis)
    axial = vol[:, :, vol.shape[2] // 2]
    # Coronal (middle slice of y-axis)
    coronal = vol[:, vol.shape[1] // 2, :]
    # Sagittal (middle slice of x-axis)
    sagittal = vol[vol.shape[0] // 2, :, :]
    return axial, coronal, sagittal

# ----------------------
# Run Feature Extraction (Process 10 Samples)
# ----------------------
# Get list of MRI files (replace with real ADNI files from GitHub)
mri_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".nii") or f.endswith(".nii.gz")]

for idx, file_name in enumerate(mri_files[:10]):  # Process first 10 samples
    file_path = os.path.join(RAW_DATA_PATH, file_name)
    
    # 1. Load raw MRI
    raw_vol = load_mri(file_path)
    
    # 2. Skull stripping
    skull_stripped_vol, brain_mask = skull_strip(raw_vol)
    
    # 3. Intensity normalization
    normalized_vol = normalize_intensity(skull_stripped_vol)
    
    # 4. Extract 3 orthogonal planes (TOP feature)
    axial, coronal, sagittal = extract_orthogonal_planes(normalized_vol)
    
    # 5. Save feature slices (for model training)
    np.save(os.path.join(FEATURE_SAVE_PATH, f"axial_{idx}.npy"), axial)
    np.save(os.path.join(FEATURE_SAVE_PATH, f"coronal_{idx}.npy"), coronal)
    np.save(os.path.join(FEATURE_SAVE_PATH, f"sagittal_{idx}.npy"), sagittal)
    
    # 6. Simulate GM/WM/CSF segmentation (thesis step)
    gm = (normalized_vol > 0.3) & (normalized_vol <= 0.6)
    wm = (normalized_vol > 0.6) & (normalized_vol <= 0.9)
    csf = (normalized_vol > 0.1) & (normalized_vol <= 0.3)
    np.save(os.path.join(ROI_SAVE_PATH, f"gm_{idx}.npy"), gm.astype(float))
    np.save(os.path.join(ROI_SAVE_PATH, f"wm_{idx}.npy"), wm.astype(float))
    np.save(os.path.join(ROI_SAVE_PATH, f"csf_{idx}.npy"), csf.astype(float))

print("✅ TOP Feature Extraction Complete!")
print(f"📂 Features saved to: {FEATURE_SAVE_PATH}")
print(f"📂 ROI (GM/WM/CSF) saved to: {ROI_SAVE_PATH}")
