
from image_utils.dwt_svd_frequence_watermark import frequence_wm_attack_and_compare
from pathlib import Path


def main():
    # Percorsi file (modifica se necessario)
    host_path = Path('files/input.png')  # Metti qui la tua immagine host
    watermark_path = Path('files/watermark.png')  # Metti qui il tuo logo/filigrana
    output_dir = Path('files/output_robustness')

    frequence_wm_attack_and_compare(host_path, watermark_path, output_dir)


if __name__ == "__main__":
    main()