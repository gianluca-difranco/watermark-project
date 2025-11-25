import os
from pathlib import Path

import cv2

from image_utils.space_domain_watermark import apply_watermark
from image_utils.utils import apply_attacks

from PIL import Image

if __name__ == '__main__':
    input_image_path = Path('files/input.png')  # Metti qui la tua immagine host
    watermark_path = Path('files/watermark.png')  # Metti qui il tuo logo/filigrana
    output_dir_path = Path("files/space_domain_watermark")

    apply_watermark(input_image_path=input_image_path,output_dir_path=output_dir_path, watermark_image_path=watermark_path)

    # --- 1. Carica l'immagine originale ---
    img_originale = Image.open("files/space_domain_watermark/bw_modified.png")
    print(f"Immagine originale caricata: {img_originale.format} {img_originale.size} {img_originale.mode}")

    watermarked_img = cv2.imread(os.path.join(str(output_dir_path), "bw_modified.png"), cv2.IMREAD_GRAYSCALE)


    # --- 2. ATTACCHI E ESTRAZIONE ---
    print("\n--- Avvio Test Robustezza ---")
    attacks = apply_attacks(watermarked_img)

    for attack_name, attacked_img in attacks.items():
        print(f"\nTestando attacco: {attack_name}")
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cv2.imwrite(f'{str(output_dir_path)}/attacked_{safe_name}.png', attacked_img)

        extracted_wm = (attacked_img & 1)* 255
        cv2.imwrite(f'{str(output_dir_path)}/extracted_from_{safe_name}.png', extracted_wm)


