import cv2
import os
from skimage.metrics import structural_similarity as ssim
from image_utils.attacks_utils import apply_attacks
from image_utils.robust_watermark import save_key
from image_utils.robust_watermark import embed_watermark_dwt_svd, extract_watermark_dwt_svd



def main():
    # Percorsi file (modifica se necessario)
    host_path = 'files/input.png'  # Metti qui la tua immagine host
    watermark_path = 'files/watermark.png'  # Metti qui il tuo logo/filigrana
    output_dir = 'files/output_robustness'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Caricamento
    # print(f"Caricamento {host_path} e {watermark_path}...")
    # host_img = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)


    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # --- 1. EMBEDDING ---
    print("\n--- Avvio Embedding DWT-SVD ---")
    alpha = 0.1  # Forza della filigrana
    watermarked_img, embedding_data = embed_watermark_dwt_svd(host_path, watermark_path, alpha=alpha)
    # 4. Salva la CHIAVE (da tenere segreta sul tuo PC)
    save_key('files/watermarked_img.npz', embedding_data)
    # Salva risultato
    cv2.imwrite(f'{output_dir}/watermarked_img.png', watermarked_img)
    print(f"Immagine watermarked salvata in {output_dir}")

    # --- 2. ATTACCHI E ESTRAZIONE ---
    print("\n--- Avvio Test Robustezza ---")
    attacks = apply_attacks(watermarked_img)

    # Ridimensioniamo il watermark originale per calcolare SSIM dopo
    wm_original_resized = cv2.resize(watermark_img,
                                     (embedding_data['shape_wm'][1], embedding_data['shape_wm'][0]))

    for attack_name, attacked_img in attacks.items():
        print(f"\nTestando attacco: {attack_name}")

        # Salva immagine attaccata
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cv2.imwrite(f'{output_dir}/attacked_{safe_name}.png', attacked_img)

        # Estrazione
        extracted_wm = extract_watermark_dwt_svd(attacked_img, key_path='files/watermarked_img.npz')
        cv2.imwrite(f'{output_dir}/extracted_from_{safe_name}.png', extracted_wm)

        # Calcolo similarità (SSIM)
        try:
            score = ssim(wm_original_resized, extracted_wm, data_range=255)
            print(f"-> Qualità Filigrana Estratta (SSIM): {score:.4f}")
        except ValueError as e:
            print(f"-> Errore calcolo SSIM (dimensioni diverse?): {e}")


if __name__ == "__main__":
    main()