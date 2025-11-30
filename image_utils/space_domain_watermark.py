import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

from classes.dataclass import SaveAttackContext
from image_utils.utils import apply_attacks, save_and_compare, calculate_ssim, calculate_psnr


#cambiare l'input con un path di un immagine
def apply_watermark(input_image_path: Path, output_dir_path: Path, watermark_image_path: Path) -> Path:
    """
    This function draw a word/phrase in image's LSB.
    """

    bw_original = cv2.imread(str(input_image_path), cv2.IMREAD_GRAYSCALE)
    # === Crea la versione originale in bianco e nero ===
    # bw_original = Image.fromarray(channel.astype(np.uint8))

    # === Estrae la matrice dei bit meno significativi (LSB) originali ===
    lsb_original = (bw_original & 1) * 255
    lsb_original_img = Image.fromarray(lsb_original.astype(np.uint8))

    # === Crea immagine con testo nero su sfondo bianco ===
    watermark_image = cv2.imread(str(watermark_image_path), cv2.IMREAD_GRAYSCALE)
    msb_plane = (watermark_image >> 7) & 1
    # === Sostituisce i LSB del canale originale con il testo ===
    channel_modified = (bw_original & 254) | msb_plane

    # === Ricrea immagine modificata e la sua versione dei LSB ===
    bw_modified = Image.fromarray(channel_modified.astype(np.uint8))
    lsb_modified = (channel_modified & 1) * 255
    lsb_modified_img = Image.fromarray(lsb_modified.astype(np.uint8))
    # === Salva i risultati ===
    os.makedirs(output_dir_path, exist_ok=True)
    cv2.imwrite(f'{output_dir_path}/bw_original.png', bw_original)
    lsb_original_img.save(f'{output_dir_path}/lsb_original.png')
    bw_modified.save(f'{output_dir_path}/watermarked_img.png')
    lsb_modified_img.save(f'{output_dir_path}/watermark.png')
    return output_dir_path / 'watermarked_img.png'


def extract_watermark(image: np.ndarray) -> np.ndarray:
    # === Estrae la matrice dei bit meno significativi (LSB) originali ===
    trans_lsb_original = (image & 1) * 255
    return trans_lsb_original.astype(np.uint8)

def space_wm_attack_and_compare(host_path: Path, watermark_path:Path, output_dir_path : Path) -> None:
    if not output_dir_path.exists():
        os.makedirs(output_dir_path)

    watermarked_img_path: Path = apply_watermark(input_image_path=host_path,output_dir_path=output_dir_path, watermark_image_path=watermark_path)

    attacks = apply_attacks(watermarked_img_path)


    context = SaveAttackContext(attacks, output_dir_path, extract_watermark)
    output_file_dict: dict[str,Path] = save_and_compare(context)
    image_watermarked = cv2.imread(str(watermarked_img_path), cv2.IMREAD_GRAYSCALE)
    for key, value in output_file_dict.items():
        if key.startswith('extracted'):
            calculate_ssim(image_watermarked, cv2.imread(str(value), cv2.IMREAD_GRAYSCALE))
            calculate_psnr(image_watermarked, cv2.imread(str(value), cv2.IMREAD_GRAYSCALE))
