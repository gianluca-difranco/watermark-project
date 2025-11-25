import os

import numpy as np
import pywt
import cv2
from pathlib import Path

from classes.dataclass import SaveAttackContext
from image_utils.utils import apply_attacks, save_and_compare, calculate_ssim


def resize_watermark(wm, target_shape):
    """Ridimensiona la filigrana per adattarla alle dimensioni della sottobanda."""
    # cv2.resize accetta (width, height), mentre shape è (rows, cols)
    return cv2.resize(wm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


# --- NUOVE FUNZIONI PER GESTIONE CHIAVE ---

def save_key(path, embedding_data):
    """Salva i dati necessari per l'estrazione in un file compresso .npz"""
    np.savez_compressed(
        path,
        U_wm=embedding_data['U_wm'],
        V_wm=embedding_data['V_wm'],
        S_LL_orig=embedding_data['S_LL_orig'],
        alpha=embedding_data['alpha'],
        shape_wm=embedding_data['shape_wm']
    )
    print(f"[INFO] Chiave di watermarking salvata in: {path}")


def load_key(path: Path):
    """Carica la chiave per l'estrazione"""
    if not path.exists():
        raise FileNotFoundError(f"Impossibile trovare il file chiave: {path}")

    data = np.load(str(path))
    embedding_data = {
        'U_wm': data['U_wm'],
        'V_wm': data['V_wm'],
        'S_LL_orig': data['S_LL_orig'],
        'alpha': float(data['alpha']),
        'shape_wm': tuple(data['shape_wm'])
    }
    return embedding_data
# --- FASE DI INCORPORAMENTO ---

def apply_watermark(input_image_path: Path, output_dir_path: Path, watermark_image_path: Path) -> Path:
    """
    Incorpora la filigrana modificando i valori singolari (SVD) della sottobanda LL (DWT).
    """

    alpha = 0.1

    host_img = cv2.imread(str(input_image_path), cv2.IMREAD_GRAYSCALE)
    watermark_img = cv2.imread(str(watermark_image_path), cv2.IMREAD_GRAYSCALE)
    if host_img is None or watermark_img is None:
        print("ERRORE: Immagini non trovate. Inserisci 'lena_512.png' e 'watermark.png' nella cartella files/.")
        return None, None

    # Pre-elaborazione
    host_img = host_img.astype(np.float32) / 255.0
    watermark_img = watermark_img.astype(np.float32) / 255.0

    # Gestione scala di grigi
    if host_img.ndim == 3:
        host_gray = cv2.cvtColor(host_img, cv2.COLOR_BGR2GRAY)
    else:
        host_gray = host_img

    # 1. DWT (Discrete Wavelet Transform)
    coeffs = pywt.dwt2(host_gray, 'haar')
    LL, (LH, HL, HH) = coeffs

    # 2. Ridimensionamento Filigrana per matchare LL
    wm_resized = resize_watermark(watermark_img, LL.shape)

    # 3. SVD sulla sottobanda LL dell'host
    U_LL, S_LL, V_LL = np.linalg.svd(LL, full_matrices=False)

    # 4. SVD sulla filigrana
    U_wm, S_wm, V_wm = np.linalg.svd(wm_resized, full_matrices=False)

    # 5. Embedding: Modifica dei valori singolari
    # Formula: S_new = S_host + alpha * S_watermark
    # S_LL e S_wm sono vettori 1D dei valori singolari
    S_LL_new = S_LL + (alpha * S_wm)

    # 6. Ricostruzione SVD -> LL modificato
    # Creiamo la matrice diagonale dai nuovi valori singolari
    Sigma_LL_new = np.diag(S_LL_new)
    LL_w = np.dot(U_LL, np.dot(Sigma_LL_new, V_LL))

    # 7. IDWT (Inverse DWT) per ottenere l'immagine watermarked
    coeffs_w = LL_w, (LH, HL, HH)
    img_w = pywt.idwt2(coeffs_w, 'haar')

    # Clipping e conversione finale
    img_w = np.clip(img_w * 255, 0, 255).astype(np.uint8)

    # --- SALVATAGGIO DATI CHIAVE PER ESTRAZIONE ---
    # Salviamo S_LL (valori originali) per poter fare la differenza inversa dopo.
    embedding_data = {
        'U_wm': U_wm,  # Vettori singolari della filigrana
        'V_wm': V_wm,  # Vettori singolari della filigrana
        'S_LL_orig': S_LL,  # Valori singolari ORIGINALI dell'host (NECESSARI)
        'alpha': alpha,
        'shape_wm': wm_resized.shape  # Utile per controlli
    }
    # 4. Salva la CHIAVE (da tenere segreta sul tuo PC)
    print(f"chiave qui {output_dir_path}/watermarked_img.npz")
    save_key(output_dir_path/'watermarked_img.npz', embedding_data)
    # Salva risultato
    cv2.imwrite(f'{output_dir_path}/watermarked_img.png', img_w)

    return output_dir_path / 'watermarked_img.png'


# --- FASE DI ESTRAZIONE ---

def extract_watermark_dwt_svd(attacked_img, embedding_data=None, key_path=None):
    """
    Estrae la filigrana dall'immagine (potenzialmente attaccata).
    Richiede embedding_data (chiave) contenente i vettori singolari della filigrana
    e i valori singolari originali dell'host.
    """
    if embedding_data is None:
        if key_path is None:
            raise ValueError("Devi fornire il percorso del file chiave (.npz) o i dati embedding_data.")
        embedding_data = load_key(key_path)

    # Recupero dati chiave
    U_wm = embedding_data['U_wm']
    V_wm = embedding_data['V_wm']
    S_LL_orig = embedding_data['S_LL_orig']
    alpha = embedding_data['alpha']

    # Pre-elaborazione immagine attaccata
    attacked_img = attacked_img.astype(np.float32) / 255.0
    if attacked_img.ndim == 3:
        attacked_gray = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY)
    else:
        attacked_gray = attacked_img

    # 1. DWT sull'immagine attaccata
    coeffs_att = pywt.dwt2(attacked_gray, 'haar')
    LL_att, _ = coeffs_att

    # 2. Resize (solo per sicurezza, se l'attacco ha cambiato le dimensioni, qui potrebbe fallire)
    # Assumiamo che l'attacco crop/resize sia stato gestito o che l'immagine sia tornata alla dim originale.
    if LL_att.shape != embedding_data['shape_wm']:
        LL_att = cv2.resize(LL_att, (embedding_data['shape_wm'][1], embedding_data['shape_wm'][0]))

    # 3. SVD sulla banda LL attaccata
    _, S_LL_att, _ = np.linalg.svd(LL_att, full_matrices=False)

    # 4. Estrazione dei valori singolari della filigrana
    # Formula inversa: S_wm_extracted = (S_attacked - S_original_host) / alpha
    # Nota: S_LL_att e S_LL_orig devono avere la stessa lunghezza.
    min_len = min(len(S_LL_att), len(S_LL_orig))
    S_wm_extracted = (S_LL_att[:min_len] - S_LL_orig[:min_len]) / alpha

    # 5. Ricostruzione della filigrana usando i vettori U_wm, V_wm salvati
    Sigma_wm_extracted = np.diag(S_wm_extracted)

    # Attenzione alle dimensioni: U_wm e V_wm devono matchare la lunghezza di S
    wm_extracted = np.dot(U_wm[:, :min_len], np.dot(Sigma_wm_extracted, V_wm[:min_len, :]))

    # Normalizzazione e conversione
    wm_extracted = np.clip(wm_extracted * 255, 0, 255).astype(np.uint8)

    return wm_extracted



def frequence_wm_attack_and_compare(host_path : Path, watermark_path : Path, output_dir_path : Path) -> None:

    if not output_dir_path.exists():
        os.makedirs(output_dir_path)

    watermark_img = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)


    watermarked_img_path = apply_watermark(host_path, output_dir_path, watermark_path)

    attacks = apply_attacks(watermarked_img_path)


    key_path: Path = output_dir_path / 'watermarked_img.npz'
    embedding_data = load_key(key_path)


    wm_original_resized = cv2.resize(watermark_img,
                                     embedding_data['shape_wm'])
    context = SaveAttackContext(attacks, key_path, output_dir_path, extract_watermark_dwt_svd)
    output_file_dict: dict[str,Path] = save_and_compare(context)

    for key, value in output_file_dict.items():
        if key.startswith('extracted'):
            calculate_ssim(wm_original_resized, cv2.imread(str(value), cv2.IMREAD_GRAYSCALE))