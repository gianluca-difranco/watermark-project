import cv2
import numpy as np
import os
import sys
from skimage.metrics import structural_similarity as ssim

# Assumiamo che il file sopra sia salvato come image_utils/space_domain_watermark.py
# Se lo hai rinominato in robust_watermark.py, cambia l'import qui sotto.
try:
    from image_utils.robust_watermark import embed_watermark_dwt_svd, extract_watermark_dwt_svd
except ImportError:
    # Fallback se esegui lo script direttamente fuori dalla struttura module
    print("Attenzione: Assicurati di essere nella root del progetto o che i percorsi siano corretti.")
    sys.exit(1)


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


def main():
    # Percorsi file (modifica se necessario)
    host_path = 'files/input.png'  # Metti qui la tua immagine host
    watermark_path = 'files/watermark.png'  # Metti qui il tuo logo/filigrana
    output_dir = 'files/output_robustness'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Caricamento
    print(f"Caricamento {host_path} e {watermark_path}...")
    host_img = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    if host_img is None or watermark_img is None:
        print("ERRORE: Immagini non trovate. Inserisci 'lena_512.png' e 'watermark.png' nella cartella files/.")
        return

    # --- 1. EMBEDDING ---
    print("\n--- Avvio Embedding DWT-SVD ---")
    alpha = 0.1  # Forza della filigrana
    watermarked_img, embedding_data = embed_watermark_dwt_svd(host_img, watermark_img, alpha=alpha)

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
        extracted_wm = extract_watermark_dwt_svd(attacked_img, embedding_data)
        cv2.imwrite(f'{output_dir}/extracted_from_{safe_name}.png', extracted_wm)

        # Calcolo similarità (SSIM)
        try:
            score = ssim(wm_original_resized, extracted_wm, data_range=255)
            print(f"-> Qualità Filigrana Estratta (SSIM): {score:.4f}")
        except ValueError as e:
            print(f"-> Errore calcolo SSIM (dimensioni diverse?): {e}")


if __name__ == "__main__":
    main()