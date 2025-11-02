# --- Creazione file di test ---
# (Esegui questo blocco una volta per creare i file necessari)
import cv2
import numpy as np
from image_utils.watermark_dwt import embed_watermark_dwt,extract_watermark_dwt
# 1. Crea un'immagine originale fittizia
img_original_arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
cv2.rectangle(img_original_arr, (100, 100), (400, 400), (128, 128, 128), -1) # Aggiungi un quadrato
cv2.imwrite("immagine_originale.png", img_original_arr)

# 2. Crea un logo fittizio
# img_logo_arr = np.zeros((100, 100), dtype=np.uint8)
# cv2.circle(img_logo_arr, (50, 50), 40, 255, -1) # Un cerchio bianco
# cv2.imwrite("logo.png", img_logo_arr)

input_image = "files/input.png"

print("Immagini di test create.")

# --- Esempio 1: Incorporare un LOGO ---

ALPHA_STRENGTH = 0.1
WAVELET_TYPE = 'haar'

print("\n--- Test 1: Incorporazione Logo ---")
embed_watermark_dwt(
    original_image_path=input_image,
    watermark_data="files/logo.png",
    output_image_path="immagine_con_logo.png",
    watermark_type='image',
    alpha=ALPHA_STRENGTH,
    wavelet=WAVELET_TYPE
)

print("\n--- Test 1: Estrazione Logo ---")
extract_watermark_dwt(
    original_image_path=input_image,
    watermarked_image_path="immagine_con_logo.png",
    output_watermark_path="logo_estratto.png",
    alpha=ALPHA_STRENGTH,
    wavelet=WAVELET_TYPE
)


# --- Esempio 2: Incorporare TESTO ---

print("\n--- Test 2: Incorporazione Testo ---")
embed_watermark_dwt(
    original_image_path=input_image,
    watermark_data="TEXT",
    output_image_path="immagine_con_testo.png",
    watermark_type='text',
    alpha=ALPHA_STRENGTH,
    wavelet=WAVELET_TYPE
)

print("\n--- Test 2: Estrazione Testo ---")
extract_watermark_dwt(
    original_image_path=input_image,
    watermarked_image_path="immagine_con_testo.png",
    output_watermark_path="testo_estratto.png",
    alpha=ALPHA_STRENGTH,
    wavelet=WAVELET_TYPE
)