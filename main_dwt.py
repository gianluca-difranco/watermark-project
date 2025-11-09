# --- Creazione file di test ---
# (Esegui questo blocco una volta per creare i file necessari)
import cv2
import numpy as np

from image_utils.frequence_domain_watermark import FrequenceDomainWatermark
from image_utils.watermark_dwt import embed_watermark_dwt,extract_watermark_dwt

from classes.watermark_mode import TEXT_MODE

ALPHA_STRENGTH = 0.1
WAVELET_TYPE = 'haar'

if __name__ == '__main__':
    fdw = FrequenceDomainWatermark(input_image="files/input.png")
    fdw.apply_watermark(watermark_data="HelloWorld!", watermark_type=TEXT_MODE)
    fdw.show_watermark()

# input_image = "files/input.png"
#
# print("\n--- Test 1: Incorporazione Logo ---")
# embed_watermark_dwt(
#     original_image_path=input_image,
#     watermark_data="files/logo.png",
#     output_image_path="immagine_con_logo.png",
#     watermark_type='image',
#     alpha=ALPHA_STRENGTH,
#     wavelet=WAVELET_TYPE
# )
#
# print("\n--- Test 1: Estrazione Logo ---")
# extract_watermark_dwt(
#     original_image_path=input_image,
#     watermarked_image_path="immagine_con_logo.png",
#     output_watermark_path="logo_estratto.png",
#     alpha=ALPHA_STRENGTH,
#     wavelet=WAVELET_TYPE
# )
#
#
# # --- Esempio 2: Incorporare TESTO ---
#
# print("\n--- Test 2: Incorporazione Testo ---")
# embed_watermark_dwt(
#     original_image_path=input_image,
#     watermark_data="TEXT",
#     output_image_path="immagine_con_testo.png",
#     watermark_type='text',
#     alpha=ALPHA_STRENGTH,
#     wavelet=WAVELET_TYPE
# )
#
# print("\n--- Test 2: Estrazione Testo ---")
# extract_watermark_dwt(
#     original_image_path=input_image,
#     watermarked_image_path="immagine_con_testo.png",
#     output_watermark_path="testo_estratto.png",
#     alpha=ALPHA_STRENGTH,
#     wavelet=WAVELET_TYPE
# )