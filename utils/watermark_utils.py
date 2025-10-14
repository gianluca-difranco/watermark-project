import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

from classes import channels, show_mode

def apply_watermark(input_path: str, output_dir: str,input_text: str = 'Hello World!', mode:str = show_mode.SHOW_MODE):
    """
    This function draw a word/phrase in image's LSB.
    :param input_path: input image path
    :param output_dir: output dir
    :param input_text: text to draw into image
    :param mode: 'show_mode' (results are only shown) or 'download_mode' (download in output_dir)
    """

    # === Carica immagine e seleziona il canale BLUE ===
    img = Image.open(input_path).convert("RGB")
    channel_index = channels.BLUE
    channel = np.array(img)[:, :, channel_index]

    # === Crea la versione originale in bianco e nero ===
    bw_original = Image.fromarray(channel.astype(np.uint8))

    # === Estrae la matrice dei bit meno significativi (LSB) originali ===
    lsb_original = (channel & 1) * 255
    lsb_original_img = Image.fromarray(lsb_original.astype(np.uint8))

    # === Crea immagine con testo nero su sfondo bianco ===
    text_img = draw_watermark_frame(img, input_text)

    # Converte in bit binari (1 per testo, 0 per sfondo)
    text_bits = np.array(text_img)
    text_bits_bin = (text_bits < 128).astype(np.uint8)

    # === Sostituisce i LSB del canale originale con il testo ===
    channel_modified = (channel & 254) | text_bits_bin

    # === Ricrea immagine modificata e la sua versione dei LSB ===
    bw_modified = Image.fromarray(channel_modified.astype(np.uint8))
    lsb_modified = (channel_modified & 1) * 255
    lsb_modified_img = Image.fromarray(lsb_modified.astype(np.uint8))



    if mode == show_mode.DOWNLOAD_MODE:
        # === Salva i risultati ===
        os.makedirs(output_dir, exist_ok=True)
        bw_original.save(os.path.join(output_dir, "bw_original.png"))
        lsb_original_img.save(os.path.join(output_dir, "lsb_original.png"))
        bw_modified.save(os.path.join(output_dir, "bw_modified.png"))
        lsb_modified_img.save(os.path.join(output_dir, "lsb_modified.png"))

        print("Tutte le immagini sono state salvate nella cartella 'files/'")
    else:
        # === Mostra le immagini affiancate ===
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 4, 1)
        plt.title("Originale (bianco e nero)")
        plt.imshow(bw_original, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("LSB originali")
        plt.imshow(lsb_original_img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Immagine con testo nei LSB")
        plt.imshow(bw_modified, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("LSB dopo inserimento testo")
        plt.imshow(lsb_modified_img, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def draw_watermark_frame(img, input_text: str):
    text_img = Image.new("L", img.size, color=255)
    draw = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    text = input_text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    x = (img.width - text_bbox[2]) // 2
    y = (img.height - text_bbox[3]) // 2
    draw.text((x, y), text, fill=0, font=font)
    return text_img
