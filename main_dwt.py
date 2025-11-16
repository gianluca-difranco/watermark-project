# --- Creazione file di test ---
# (Esegui questo blocco una volta per creare i file necessari)
import cv2
import numpy as np

from PIL import Image
# Importa le funzioni che abbiamo appena scritto (assumendo che il file si chiami trasformazioni.py)
from image_utils.transformation import resize_image, rotate_image, crop_image, flip_image, adjust_brightness, apply_blur

from image_utils.frequence_domain_watermark import FrequenceDomainWatermark
from image_utils.watermark_dwt import embed_watermark_dwt,extract_watermark_dwt

from classes.watermark_mode import TEXT_MODE

ALPHA_STRENGTH = 0.1
WAVELET_TYPE = 'haar'

if __name__ == '__main__':
    fdw = FrequenceDomainWatermark(input_image="files/input.png")
    fdw.apply_watermark(watermark_data="HelloWorld!", watermark_type=TEXT_MODE)



    # --- 1. Carica l'immagine originale ---
    img_originale = Image.open("files/watermarked_output.png")
    print(f"Immagine originale caricata: {img_originale.format} {img_originale.size} {img_originale.mode}")

    # Esempio di come creare l'immagine per il test
    img_wm = cv2.imread("files/watermarked_output.png")
    # Salva come JPEG con qualità 80 (su 100)
    cv2.imwrite("files/immagine_compressa.jpg", img_wm, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # --- 2. Applica una catena di trasformazioni ---

    # Nota: applichiamo le trasformazioni in sequenza,
    # passando il risultato della funzione precedente a quella successiva.

    print("Applico trasformazioni...")

    # a. Ridimensiona a una larghezza di 500px mantenendo le proporzioni
    #img_temp_1 = resize_image(img_originale, new_width=200, keep_aspect_ratio=True)
    #print(f"Dimensione dopo resize: {img_temp_1.size}")

    # b. Ruota l'immagine di 15 gradi (espandendo per non tagliare i bordi)
    #img_temp_2 = rotate_image(img_originale, angle=15, fill_color=(0, 0, 0))  # Riempi di nero
    #
    # # c. Ritaglia una regione centrale
    # # Per centrare, calcoliamo il box
    # w, h = img_originale.size
    # box = (w / 4, h / 4, w * 3 / 4, h * 3 / 4)  # Prendi il 50% centrale
    # img_temp_3 = crop_image(img_originale, box)
    # # print(f"Dimensione dopo crop: {img_temp_3.size}")
    # #
    # # # d. Specchia l'immagine orizzontalmente
    # # img_temp_4 = flip_image(img_temp_3, mode='horizontal')
    # #
    # # # e. Aumenta la luminosità del 20%
    # # img_finale = adjust_brightness(img_temp_4, factor=1.2)
    # #
    # # # --- 3. Salva e mostra il risultato ---
    # output_path = "files/watermarked_transformed_output.png"
    # img_temp_3.save(output_path)
    # print(f"Immagine trasformata salvata in: {output_path}")

    # Apre l'immagine nel visualizzatore di default del tuo OS
    #img_finale.show()
    fdw.show_watermark(transfomed_img_path="files/immagine_compressa.jpg")

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