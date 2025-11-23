import os

import cv2

from image_utils.space_domain_watermark import SpaceDomainWatermark
from image_utils.space_test import LSBWatermarker
from image_utils.attacks_utils import apply_attacks

from PIL import Image
# Importa le funzioni che abbiamo appena scritto (assumendo che il file si chiami trasformazioni.py)
from image_utils.transformation import resize_image, rotate_image, crop_image, flip_image, adjust_brightness, apply_blur
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # domain_watermark = SpaceDomainWatermark(input_image="files/input.png")
    # domain_watermark.apply_watermark(input_text="HelloWorld!")
    # domain_watermark.show_watermark()
    host_path = 'files/input.png'  # Metti qui la tua immagine host
    watermark_path = 'files/watermark.png'  # Metti qui il tuo logo/filigrana
    output_dir = "files/space_domain_watermark"

    marker = SpaceDomainWatermark(host_path, output_dir=output_dir)
    marker.apply_watermark(watermark_image_path=watermark_path)
    #marker.show_watermark()



    # --- 1. Carica l'immagine originale ---
    img_originale = Image.open("files/space_domain_watermark/bw_modified.png")
    print(f"Immagine originale caricata: {img_originale.format} {img_originale.size} {img_originale.mode}")

    watermarked_img = cv2.imread(os.path.join(output_dir, "bw_modified.png"), cv2.IMREAD_GRAYSCALE)


    # --- 2. ATTACCHI E ESTRAZIONE ---
    print("\n--- Avvio Test Robustezza ---")
    attacks = apply_attacks(watermarked_img)

    for attack_name, attacked_img in attacks.items():
        print(f"\nTestando attacco: {attack_name}")
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cv2.imwrite(f'{output_dir}/attacked_{safe_name}.png', attacked_img)

        extracted_wm = (attacked_img & 1)* 255
        cv2.imwrite(f'{output_dir}/extracted_from_{safe_name}.png', extracted_wm)


