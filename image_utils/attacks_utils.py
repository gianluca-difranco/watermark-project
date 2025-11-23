import cv2
import numpy as np
from image_utils.robust_watermark import extract_watermark_dwt_svd



def apply_attacks(image):
    """Applica una serie di attacchi per testare la robustezza."""
    attacks = {}

    # 1. Compressione JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Qualità 50 (media)
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    img_jpeg = cv2.imdecode(encimg, 0)  # 0 = load as grayscale
    attacks['JPEG (Q=50)'] = img_jpeg

    # 2. Rumore Sale e Pepe
    noise = np.zeros(image.shape, np.uint8)
    cv2.randu(noise, 0, 255)
    img_noise = image.copy()
    img_noise[noise < 10] = 0  # Pepper
    img_noise[noise > 245] = 255  # Salt
    attacks['Salt & Pepper'] = img_noise

    # 3. Rotazione (Leggera) + Crop automatico (simulato dal resize implicito in visualizzazione)
    # Nota: La rotazione pesante disallinea i pixel per la DWT.
    # Senza un algoritmo di riallineamento, SVD resiste solo a piccole rotazioni.
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)  # 2 gradi
    img_rot = cv2.warpAffine(image, M, (cols, rows))
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