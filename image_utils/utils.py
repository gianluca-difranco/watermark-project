from numpy import ndarray
from pathlib import Path
import math
from classes.dataclass import SaveAttackContext
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def apply_attacks(watermarked_img_path: Path) -> dict[str,cv2.typing.MatLike]:
    """Applica una serie di attacchi per testare la robustezza."""
    attacks = {}
    watermark_img = cv2.imread(str(watermarked_img_path))

    # 1. Compressione JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Qualità 50 (media)
    _, encimg = cv2.imencode('.jpg', watermark_img, encode_param)

    img_jpeg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    attacks['JPEG (Q=50)'] = img_jpeg

    # 2. Rumore Sale e Pepe
    rows, cols = watermark_img.shape[:2]
    noise = np.zeros((rows, cols), np.uint8)

    cv2.randu(noise, 0, 255)
    img_noise = watermark_img.copy()
    img_noise[noise < 10] = 0  # Pepper
    img_noise[noise > 245] = 255  # Salt
    attacks['Salt & Pepper'] = img_noise

    # 3. Rotazione (Leggera) + Crop automatico (simulato dal resize implicito in visualizzazione)
    # Nota: La rotazione pesante disallinea i pixel per la DWT.
    # Senza un algoritmo di riallineamento, SVD resiste solo a piccole rotazioni.
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)  # 2 gradi
    img_rot = cv2.warpAffine(watermark_img, M, (cols, rows))
    attacks['Rotation (2 deg)'] = img_rot

    return attacks


def save_and_compare(context : SaveAttackContext) -> dict[str,Path]:

    output_file_dict : dict[str,Path] = {}
    for attack_name, attacked_img in context.attacks.items():
        print(f"\nTestando attacco: {attack_name}")
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        attacked_path: Path = Path(f'{context.output_dir}/attacked_{safe_name}.png')
        extracted_wm_path: Path = Path(f'{context.output_dir}/extracted_from_{safe_name}.png')


        cv2.imwrite(str(attacked_path), attacked_img)
        output_file_dict[f'attacked_{safe_name}.png'] = attacked_path

        extracted_wm = context.extract_function(attacked_img, **context.extract_parameters)
        cv2.imwrite(str(extracted_wm_path), extracted_wm)

        output_file_dict[f'extracted_from_{safe_name}.png'] = extracted_wm_path


    return output_file_dict


def calculate_ssim(img1: ndarray, img2: ndarray) -> float:
    try:
        score = ssim(img1, img2, data_range=255)
        print(f"-> Qualità Filigrana Estratta (SSIM): {score:.4f}")
    except ValueError as e:
        print(f"-> Errore calcolo SSIM (dimensioni diverse?): {e}")
        score = 0.0

    return score


def calculate_mse(imageA, imageB):
    """
    Mean Squared Error: Calcola la media degli errori quadrati tra i pixel corrispondenti.
    Un valore più basso è migliore.
    """
    # Assicurarsi che le immagini siano float per evitare overflow
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print(f"MSE: {err:.4f}")
    return err


def calculate_psnr(imageA, imageB):
    """
    Peak Signal-to-Noise Ratio: Misura il rapporto tra la potenza massima del segnale e il rumore.
    Valori sopra i 30dB indicano solitamente un'ottima qualità invisibile.
    """
    mse = calculate_mse(imageA, imageB)

    # Se le immagini sono identiche, MSE è 0 e PSNR è infinito
    if mse == 0:
        return 100

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    print(f"PSNR: {psnr:.4f}")
    return psnr