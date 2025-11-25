import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from pathlib import Path


#cambiare l'input con un path di un immagine
def apply_watermark(input_image_path: Path, output_dir_path: Path, watermark_image_path: Path) -> None:
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
    bw_modified.save(f'{output_dir_path}/bw_modified.png')
    lsb_modified_img.save(f'{output_dir_path}/watermark.png')


def show_watermark(self, transfomed_img_path=None):

    bw_original = Image.open(os.path.join(self.output_dir, "bw_original.png"))
    lsb_original_img= Image.open(os.path.join(self.output_dir, "lsb_original.png"))
    bw_modified = Image.open(os.path.join(self.output_dir, "bw_modified.png"))
    lsb_modified_img = Image.open(os.path.join(self.output_dir, "watermark.png"))
    if transfomed_img_path:
        transformed_img = Image.open(transfomed_img_path)
        channel = np.array(transformed_img)[:, :]


        # === Estrae la matrice dei bit meno significativi (LSB) originali ===
        trans_lsb_original = (channel & 1) * 255
        trans_lsb_original_img = Image.fromarray(trans_lsb_original.astype(np.uint8))

    # === Mostra le immagini affiancate ===
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 4, 1)
    plt.title("Originale (bianco e nero)")
    plt.imshow(bw_original, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("LSB originali")
    plt.imshow(lsb_original_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Immagine con testo nei LSB")
    plt.imshow(bw_modified, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title("LSB dopo inserimento testo")
    plt.imshow(lsb_modified_img, cmap="gray")

    if transfomed_img_path:
        plt.subplot(1, 4, 4)
        plt.title("Trasformata")
        plt.imshow(trans_lsb_original_img)

    plt.axis("off")
    plt.tight_layout()
    plt.show()



def _draw_watermark_frame(self, img, input_text: str):
    text_img = Image.new("L", img.size, color=0)
    draw = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    text = input_text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    x = (img.width - text_bbox[2]) // 2
    y = (img.height - text_bbox[3]) // 2
    draw.text((x, y), text, fill=255, font=font)
    return text_img


def _generate_text_watermark_image(self, width, height, text_to_repeat):
    """
    Generates a black and white image (1-bit depth) filled with repeated text.
    The text will be white on a black background, representing the LSB bits.
    """
    # Create a black background image
    text_img = Image.new('1', (width, height), color=0)  # '1' for 1-bit pixels (black/white)
    draw = ImageDraw.Draw(text_img)

    # Try to find a system font or use a default
    try:
        # Adjust font size dynamically to make text span almost entire image
        font_size = 1  # Start small
        font = ImageFont.truetype("arial.ttf", font_size)  # Default font, try common one

        # Find a suitable font size that allows text to be visible but also fills
        while True:
            temp_font = ImageFont.truetype("arial.ttf", font_size)  # Or a path to your font
            text_bbox = draw.textbbox((0, 0), text_to_repeat, font=temp_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Check if the text fits at least once or is too big already
            if text_width < width / 5 and text_height < height / 5:  # Arbitrary factor to ensure multiple repeats
                font_size += 1
            else:
                font = temp_font
                break
            if font_size > min(width, height):  # Prevent infinite loop for very small images/large text
                break
    except IOError:
        print("Warning: Arial font not found, using default PIL font. Text might look different.", file=sys.stderr)
        font = ImageFont.load_default()
        # If default font, calculate size based on bounding box
        text_bbox = draw.textbbox((0, 0), text_to_repeat, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except Exception as e:
        print(f"Error loading font: {e}, using default PIL font.", file=sys.stderr)
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text_to_repeat, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

    # If text_width or text_height is 0 (e.g., empty string), set to a default to avoid division by zero
    if text_width == 0: text_width = 100
    if text_height == 0: text_height = 50

    # Draw the text repeatedly
    for y in range(0, height + text_height, text_height):
        for x in range(0, width + text_width, text_width):
            draw.text((x, y), text_to_repeat, font=font, fill=1)  # Fill with 1 (white)

    return text_img
