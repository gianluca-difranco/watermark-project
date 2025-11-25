import cv2
import numpy as np
from image_utils.dwt_svd_frequence_watermark import extract_watermark_dwt_svd
from pathlib import Path


def apply_attacks(watermarked_img_path: Path) -> dict[str,cv2.typing.MatLike]:
    """Applica una serie di attacchi per testare la robustezza."""
    attacks = {}
    watermark_img = cv2.imread(str(watermarked_img_path), cv2.IMREAD_GRAYSCALE)

    # 1. Compressione JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Qualità 50 (media)
    _, encimg = cv2.imencode('.jpg', watermark_img, encode_param)
    img_jpeg = cv2.imdecode(encimg, 0)  # 0 = load as grayscale
    attacks['JPEG (Q=50)'] = img_jpeg

    # 2. Rumore Sale e Pepe
    noise = np.zeros(watermark_img.shape, np.uint8)
    cv2.randu(noise, 0, 255)
    img_noise = watermark_img.copy()
    img_noise[noise < 10] = 0  # Pepper
    img_noise[noise > 245] = 255  # Salt
    attacks['Salt & Pepper'] = img_noise

    # 3. Rotazione (Leggera) + Crop automatico (simulato dal resize implicito in visualizzazione)
    # Nota: La rotazione pesante disallinea i pixel per la DWT.
    # Senza un algoritmo di riallineamento, SVD resiste solo a piccole rotazioni.
    rows, cols = watermark_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)  # 2 gradi
    img_rot = cv2.warpAffine(watermark_img, M, (cols, rows))
    attacks['Rotation (2 deg)'] = img_rot

    return attacks


def save_attacks(attacks: dict[str,cv2.typing.MatLike], output_dir):
    for attack_name, attacked_img in attacks.items():
        print(f"\nTestando attacco: {attack_name}")

        # Salva immagine attaccata
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cv2.imwrite(f'{output_dir}/attacked_{safe_name}.png', attacked_img)

        # Estrazione
        extracted_wm = extract_watermark_dwt_svd(attacked_img, embedding_data)
        cv2.imwrite(f'{output_dir}/extracted_from_{safe_name}.png', extracted_wm)