import cv2
import os
from skimage.metrics import structural_similarity as ssim
from image_utils.attacks_utils import apply_attacks
from image_utils.dwt_svd_frequence_watermark import save_key, load_key
from image_utils.dwt_svd_frequence_watermark import apply_watermark, extract_watermark_dwt_svd
from pathlib import Path


def main():
    # Percorsi file (modifica se necessario)
    host_path = Path('files/input.png')  # Metti qui la tua immagine host
    watermark_path = Path('files/watermark.png')  # Metti qui il tuo logo/filigrana
    output_dir = Path('files/output_robustness')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    watermark_img = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)

    # --- 1. EMBEDDING ---
    print("\n--- Avvio Embedding DWT-SVD ---")
    watermarked_img_path = apply_watermark(host_path, output_dir, watermark_path)
    print(f"Immagine watermarked salvata in {output_dir}")

    # --- 2. ATTACCHI E ESTRAZIONE ---
    print("\n--- Avvio Test Robustezza ---")
    attacks = apply_attacks(watermarked_img_path)

    key_path = 'files\output_robustness\watermarked_img.npz'
    embedding_data = load_key(key_path)
    # Ridimensioniamo il watermark originale per calcolare SSIM dopo
    wm_original_resized = cv2.resize(watermark_img,
                                     (embedding_data['shape_wm'][1], embedding_data['shape_wm'][0]))

    for attack_name, attacked_img in attacks.items():
        print(f"\nTestando attacco: {attack_name}")

        # Salva immagine attaccata
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cv2.imwrite(f'{output_dir}/attacked_{safe_name}.png', attacked_img)

        # Estrazione
        extracted_wm = extract_watermark_dwt_svd(attacked_img, key_path=key_path)
        cv2.imwrite(f'{output_dir}/extracted_from_{safe_name}.png', extracted_wm)

        # Calcolo similarità (SSIM)
        try:
            score = ssim(wm_original_resized, extracted_wm, data_range=255)
            print(f"-> Qualità Filigrana Estratta (SSIM): {score:.4f}")
        except ValueError as e:
            print(f"-> Errore calcolo SSIM (dimensioni diverse?): {e}")


if __name__ == "__main__":
    main()