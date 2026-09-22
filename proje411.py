import pydicom
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.segmentation import chan_vese
from skimage.exposure import equalize_adapthist
import os

# DICOM dosya yolları
acute_path = r"C:\Users\silae\Desktop\yeni seçilmişler\1\ataklı_ince\153-634\1.2.840.113704.1.111.2172.1642630303.106982.dcm"
healthy_path = r"C:\Users\silae\Desktop\yeni seçilmişler\1\normal_ince\326-376\1.2.392.200036.9123.100.11.15224860852749249173983707701769187.dcm"


def read_dicom_as_hu(path):
    ds = pydicom.dcmread(path)
    img_raw = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img_raw * slope + intercept


def window_and_normalize_hu(img_hu, hu_min=-200, hu_max=300):
    img = np.clip(img_hu, hu_min, hu_max)
    return (img - hu_min) / (hu_max - hu_min)


def gamma_correction(img, gamma=0.8):
    return np.power(img, gamma)


def histogram_equalization(img):
    img_8bit = (img * 255).astype(np.uint8)
    eq = cv2.equalizeHist(img_8bit)
    return eq.astype(np.float32) / 255.0


def sharpen_image(img):
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]])
    sharp = cv2.filter2D(img.astype(np.float32), -1, kernel)
    sharp = np.clip(sharp, 0, 1)
    return sharp


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_8bit = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(img_8bit)
    return result.astype(np.float32) / 255.0


def apply_chan_vese_segmentation(img):
    seg = chan_vese(
        img,
        mu=0.25,
        lambda1=1.0,
        lambda2=1.0,
        tol=1e-3,
        max_num_iter=200,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=False
    )
    return seg


# Otsu Thresholding 

def apply_otsu_segmentation(img):
    img_8bit = (img * 255).astype(np.uint8)
    _, seg = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (seg / 255).astype(np.uint8)



# K-Means Clustering

def apply_kmeans_segmentation(img, k=3):
    pixels = img.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    seg = labels.reshape(img.shape)
    return seg.astype(np.uint8), centers.flatten()




img_acute = read_dicom_as_hu(acute_path)
img_healthy = read_dicom_as_hu(healthy_path)

img_acute_norm = window_and_normalize_hu(img_acute)
img_healthy_norm = window_and_normalize_hu(img_healthy)

print("Images loaded successfully.")
print("Healthy image shape:", img_healthy_norm.shape)
print("Acute image shape:", img_acute_norm.shape)


# 1. Gaussian + Median
acute_gauss_median = ndi.median_filter(
    gaussian(img_acute_norm, sigma=1, preserve_range=True),
    size=(3, 3)
)

healthy_gauss_median = ndi.median_filter(
    gaussian(img_healthy_norm, sigma=1, preserve_range=True),
    size=(3, 3)
)


# 2. Bilateral Filter
acute_bilateral = cv2.bilateralFilter(
    img_acute_norm.astype(np.float32), 5, 0.1, 5
)

healthy_bilateral = cv2.bilateralFilter(
    img_healthy_norm.astype(np.float32), 5, 0.1, 5
)


# 3. Bilateral + Gamma Correction
acute_bilateral_gamma = gamma_correction(acute_bilateral, gamma=0.8)
healthy_bilateral_gamma = gamma_correction(healthy_bilateral, gamma=0.8)


# 4. Bilateral + Histogram Equalization
acute_bilateral_histeq = histogram_equalization(acute_bilateral)
healthy_bilateral_histeq = histogram_equalization(healthy_bilateral)


# 5. Bilateral + Sharpening
acute_bilateral_sharp = sharpen_image(acute_bilateral)
healthy_bilateral_sharp = sharpen_image(healthy_bilateral)


# 6. Bilateral + CLAHE
acute_bilateral_clahe = apply_clahe(acute_bilateral)
healthy_bilateral_clahe = apply_clahe(healthy_bilateral)

print("\nCLAHE outputs created successfully.")
print("Healthy CLAHE min/max:", healthy_bilateral_clahe.min(), healthy_bilateral_clahe.max())
print("Acute CLAHE min/max:", acute_bilateral_clahe.min(), acute_bilateral_clahe.max())


# 7. SEGMENTATION (Chan-Vese)
healthy_bilateral_seg = apply_chan_vese_segmentation(healthy_bilateral)
acute_bilateral_seg = apply_chan_vese_segmentation(acute_bilateral)

healthy_clahe_seg = apply_chan_vese_segmentation(healthy_bilateral_clahe)
acute_clahe_seg = apply_chan_vese_segmentation(acute_bilateral_clahe)

print("\nChan-Vese Segmentation completed successfully.")
print("Healthy bilateral segmented pixels:", np.sum(healthy_bilateral_seg))
print("Acute bilateral segmented pixels:", np.sum(acute_bilateral_seg))
print("Healthy CLAHE segmented pixels:", np.sum(healthy_clahe_seg))
print("Acute CLAHE segmented pixels:", np.sum(acute_clahe_seg))


# OTSU THRESHOLDING

healthy_otsu = apply_otsu_segmentation(healthy_bilateral_clahe)
acute_otsu = apply_otsu_segmentation(acute_bilateral_clahe)

print("\n[NEW] Otsu Thresholding completed.")
print("Healthy Otsu segmented pixels:", np.sum(healthy_otsu))
print("Acute Otsu segmented pixels:", np.sum(acute_otsu))


# K-MEANS CLUSTERING (k=3)
healthy_kmeans, healthy_centers = apply_kmeans_segmentation(healthy_bilateral_clahe, k=4)
acute_kmeans, acute_centers = apply_kmeans_segmentation(acute_bilateral_clahe, k=4)

print("\n[NEW] K-Means Clustering completed.")
print("Healthy cluster centers:", np.sort(healthy_centers))
print("Acute cluster centers:", np.sort(acute_centers))


# ANA KARŞILAŞTIRMA 
plt.figure(figsize=(24, 8))

# HEALTHY
plt.subplot(2, 7, 1)
plt.imshow(img_healthy_norm, cmap='gray')
plt.title("H-Original", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 2)
plt.imshow(healthy_gauss_median, cmap='gray')
plt.title("Healthy - G+M", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 3)
plt.imshow(healthy_bilateral, cmap='gray')
plt.title("Healthy - Bilateral", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 4)
plt.imshow(healthy_bilateral_gamma, cmap='gray')
plt.title("Healthy - Bilat+Gamma", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 5)
plt.imshow(healthy_bilateral_histeq, cmap='gray')
plt.title("Healthy - Bilateral+HistEq", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 6)
plt.imshow(healthy_bilateral_sharp, cmap='gray')
plt.title("Healthy - Bilateral+Sharp", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 7)
plt.imshow(healthy_bilateral_clahe, cmap='gray')
plt.title("Healthy - Bilateral+CLAHE", fontsize=10)
plt.axis("off")


# ACUTE
plt.subplot(2, 7, 8)
plt.imshow(img_acute_norm, cmap='gray')
plt.title("A-Original", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 9)
plt.imshow(acute_gauss_median, cmap='gray')
plt.title("Acute - G+M", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 10)
plt.imshow(acute_bilateral, cmap='gray')
plt.title("Acute - Bilateral", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 11)
plt.imshow(acute_bilateral_gamma, cmap='gray')
plt.title("Acute - Bilat+Gamma", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 12)
plt.imshow(acute_bilateral_histeq, cmap='gray')
plt.title("Acute - Bilateral+HistEq", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 13)
plt.imshow(acute_bilateral_sharp, cmap='gray')
plt.title("Acute - Bilateral+Sharp", fontsize=10)
plt.axis("off")

plt.subplot(2, 7, 14)
plt.imshow(acute_bilateral_clahe, cmap='gray')
plt.title("Acute - Bi+CLAHE", fontsize=10)
plt.axis("off")

plt.subplots_adjust(wspace=0.15, hspace=0.35)
plt.tight_layout()
plt.savefig("all_methods_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# CLAHE AYRI FIGURE
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(healthy_bilateral, cmap='gray')
plt.title("Healthy - Bilateral")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(healthy_bilateral_clahe, cmap='gray')
plt.title("Healthy - Bilateral + CLAHE")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(acute_bilateral, cmap='gray')
plt.title("Acute - Bilateral")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(acute_bilateral_clahe, cmap='gray')
plt.title("Acute - Bilateral + CLAHE")
plt.axis("off")

plt.tight_layout()
plt.savefig("clahe_results.png", dpi=300, bbox_inches="tight")
plt.show()


# SEGMENTATION MASKELERİ AYRI FIGURE
plt.figure(figsize=(12, 10))

plt.subplot(2, 4, 1)
plt.imshow(healthy_bilateral, cmap="gray")
plt.title("Healthy - Bilateral")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(healthy_bilateral_seg, cmap="gray")
plt.title("Healthy - Bilateral Seg")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(healthy_bilateral_clahe, cmap="gray")
plt.title("Healthy - Bilateral+CLAHE")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(healthy_clahe_seg, cmap="gray")
plt.title("Healthy - CLAHE Seg")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(acute_bilateral, cmap="gray")
plt.title("Acute - Bilateral")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(acute_bilateral_seg, cmap="gray")
plt.title("Acute - Bilateral Seg")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(acute_bilateral_clahe, cmap="gray")
plt.title("Acute - Bilateral+CLAHE")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(acute_clahe_seg, cmap="gray")
plt.title("Acute - CLAHE Seg")
plt.axis("off")

plt.tight_layout()
plt.savefig("segmentation_masks.png", dpi=300, bbox_inches="tight")
plt.show()


# KONTUR GÖSTERİMİ
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(healthy_bilateral, cmap="gray")
plt.contour(healthy_bilateral_seg, colors="r", linewidths=1)
plt.title("Healthy - Bilateral + Seg Contour")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(healthy_bilateral_clahe, cmap="gray")
plt.contour(healthy_clahe_seg, colors="r", linewidths=1)
plt.title("Healthy - CLAHE + Seg Contour")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(acute_bilateral, cmap="gray")
plt.contour(acute_bilateral_seg, colors="r", linewidths=1)
plt.title("Acute - Bilateral + Seg Contour")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(acute_bilateral_clahe, cmap="gray")
plt.contour(acute_clahe_seg, colors="r", linewidths=1)
plt.title("Acute - CLAHE + Seg Contour")
plt.axis("off")

plt.tight_layout()
plt.savefig("segmentation_contours.png", dpi=300, bbox_inches="tight")
plt.show()


# [NEW] OTSU + K-MEANS SEGMENTASYON KARŞILAŞTIRMA FIGURE
# Chan-Vese (existing) vs Otsu (new) vs K-Means (new)

plt.figure(figsize=(16, 8))

# Row 1: Healthy
plt.subplot(2, 4, 1)
plt.imshow(healthy_bilateral_clahe, cmap='gray')
plt.title("Healthy - Input (CLAHE)")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(healthy_clahe_seg, cmap='gray')
plt.title("Healthy - Chan-Vese")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(healthy_otsu, cmap='gray')
plt.title("Healthy - Otsu [NEW]")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(healthy_kmeans, cmap='viridis')
plt.title("Healthy - K-Means [NEW]")
plt.axis("off")

# Row 2: Acute
plt.subplot(2, 4, 5)
plt.imshow(acute_bilateral_clahe, cmap='gray')
plt.title("Acute - Input (CLAHE)")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(acute_clahe_seg, cmap='gray')
plt.title("Acute - Chan-Vese")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(acute_otsu, cmap='gray')
plt.title("Acute - Otsu [NEW]")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(acute_kmeans, cmap='viridis')
plt.title("Acute - K-Means [NEW]")
plt.axis("off")

plt.suptitle("Segmentation Comparison: Chan-Vese vs Otsu vs K-Means", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("segmentation_all_methods.png", dpi=300, bbox_inches="tight")
plt.show()
