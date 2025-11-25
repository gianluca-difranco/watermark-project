
from image_utils.dwt_svd_frequence_watermark import frequence_wm_attack_and_compare
from pathlib import Path

from image_utils.space_domain_watermark import space_wm_attack_and_compare


def main():
    # Percorsi file (modifica se necessario)
    input_image_path = Path('files/input.png')
    watermark_path = Path('files/watermark.png')

    output_dir_path = Path("files/space_domain_watermark")

    #test sul dominio spaziale
    space_wm_attack_and_compare(input_image_path, watermark_path, output_dir_path)

    output_dir = Path('files/output_robustness')

    #test sul dominio delle frequenze
    frequence_wm_attack_and_compare(input_image_path, watermark_path, output_dir)


if __name__ == "__main__":
    main()