import os

import cv2
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from classes.watermark_mode import IMAGE_MODE, TEXT_MODE

from classes import channels, show_mode

class FrequenceDomainWatermark:


    def __init__(self, input_image: str, output_dir: str = "files/frequence_domain_watermark"):
        self.input_image = input_image
        self.output_dir = output_dir

    def apply_watermark(self, watermark_data,
                            watermark_type='image', alpha=0.1, wavelet='haar'):
        """
        Incorpora un watermark (logo o testo) in un'immagine utilizzando DWT.

        :param watermark_data: Path del file logo (se type='image') o stringa di testo (se type='text').
        :param watermark_type: 'image' o 'text'.
        :param alpha: Forza del watermark (valore piccolo, es. 0.05 - 0.2).
        :param wavelet: Tipo di wavelet da usare (es. 'haar', 'db1').
        """

        # 1. Carica l'immagine originale
        img = cv2.imread(self.input_image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Errore: Immagine originale non trovata in {self.input_image}")
            return

        # 2. Converte in YCrCb e separa i canali
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)

        # 3. Applica DWT al canale di luminanza (Y)
        coeffs_y = pywt.dwt2(y, wavelet)
        LL, (LH, HL, HH) = coeffs_y

        # 4. Prepara il watermark (logo o testo)
        target_shape = HH.shape  # Il watermark deve avere la dimensione della banda HH

        if watermark_type == IMAGE_MODE:
            watermark = self._prepare_logo_watermark(watermark_data, target_shape)
        elif watermark_type == TEXT_MODE:
            watermark = self._create_text_watermark(watermark_data, target_shape)
        else:
            print("Errore: watermark_type deve essere 'image' o 'text'")
            return

        if watermark is None:
            print("Errore: impossibile creare o caricare il watermark.")
            return

        # 5. Incorpora il watermark nella banda HH (High-High)
        # Formula: HH_watermarked = HH + alpha * watermark
        HH_w = HH + alpha * watermark

        # 6. Ricostruisci il canale Y
        coeffs_Y_w = (LL, (LH, HL, HH_w))
        Y_w = pywt.idwt2(coeffs_Y_w, wavelet)

        # 7. Adatta la dimensione (IDWT può causare un +1 pixel)
        Y_w = cv2.resize(Y_w, (y.shape[1], y.shape[0]))

        # 8. Assicura che i valori siano nell'intervallo [0, 255]
        Y_w = np.clip(Y_w, 0, 255)

        # 9. Unisci i canali e riconverti in BGR
        img_w_ycrcb = cv2.merge((Y_w.astype('uint8'), cr, cb))
        img_w = cv2.cvtColor(img_w_ycrcb, cv2.COLOR_YCrCb2BGR)

        # 10. Salva l'immagine finale
        watermarked_image_path = os.path.join(self.output_dir, "watermarked.png")
        cv2.imwrite(watermarked_image_path, img_w)
        print(f"Watermark incorporato con successo e salvato in: {watermarked_image_path}")
        os.makedirs(self.output_dir, exist_ok=True)
        self._extract_watermark_dwt(watermarked_image_path, os.path.join(self.output_dir, "watermark.png"))


    def show_watermark(self):
        original_image = Image.open(self.input_image)
        watermarked_image = Image.open(os.path.join(self.output_dir, "watermarked.png"))
        wm_image = Image.open(os.path.join(self.output_dir, "watermark.png"))

        # === Mostra le immagini affiancate ===
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 4, 1)
        plt.title("Originale")
        plt.imshow(original_image)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Immagine con Watermark")
        plt.imshow(watermarked_image)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Watermark")
        plt.imshow(wm_image,)
        plt.axis("off")

        plt.tight_layout()
        plt.show()





    def _extract_watermark_dwt(self, watermarked_image_path,
                              output_watermark_path, alpha=0.1, wavelet='haar'):
        """
        Estrae un watermark da un'immagine (metodo NON-BLIND).

        :param watermarked_image_path: Path dell'immagine con watermark.
        :param output_watermark_path: Path dove salvare il watermark estratto.
        :param alpha: La STESSA forza usata durante l'incorporazione.
        :param wavelet: La STESSA wavelet usata durante l'incorporazione.
        """

        # 1. Carica entrambe le immagini
        img_orig = cv2.imread(self.input_image, cv2.IMREAD_COLOR)
        img_wm = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)

        if img_orig is None or img_wm is None:
            print("Errore: impossibile caricare una o entrambe le immagini.")
            return

        # 2. Converte entrambe in YCrCb e ottieni i canali Y
        Y_orig, _, _ = cv2.split(cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb))
        Y_wm, _, _ = cv2.split(cv2.cvtColor(img_wm, cv2.COLOR_BGR2YCrCb))

        # 3. Applica DWT a entrambi i canali Y
        coeffs_Y_orig = pywt.dwt2(Y_orig, wavelet)
        _, (_, _, HH_orig) = coeffs_Y_orig

        coeffs_Y_wm = pywt.dwt2(Y_wm, wavelet)
        _, (_, _, HH_wm) = coeffs_Y_wm

        # 4. Estrai il watermark invertendo la formula
        # Formula: watermark = (HH_watermarked - HH) / alpha
        watermark_extracted_raw = (HH_wm - HH_orig) / alpha

        # 5. Post-processa il watermark estratto per renderlo visibile

        # Normalizza l'immagine estratta per visualizzarla (0-255)
        cv2.normalize(watermark_extracted_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        watermark_visible = watermark_extracted_raw.astype('uint8')

        # Binarizza per pulire il rumore
        _, watermark_final = cv2.threshold(watermark_visible, 127, 255, cv2.THRESH_BINARY)

        # 6. Salva il watermark estratto
        cv2.imwrite(output_watermark_path, watermark_final)
        print(f"Watermark estratto e salvato in: {output_watermark_path}")
        return watermark_final



    def _create_text_watermark(self, text, target_shape):
        """
        Crea un'immagine in scala di grigi da una stringa di testo,
        dimensionata per il sub-band DWT di destinazione.
        """
        try:
            # Crea un'immagine PIL nera
            img_pil = Image.new('L', (target_shape[1], target_shape[0]), 0)  # (width, height)
            draw = ImageDraw.Draw(img_pil)

            # Prova a caricare un font
            try:
                # Stima una dimensione del font
                font_size = int(target_shape[0] / 4)
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                print("Arial.ttf non trovato, uso il font di default.")
                # Se arial non è disponibile, usa il default
                font = ImageFont.load_default()

            # Calcola la posizione per centrare il testo
            try:
                # Metodo moderno PIL
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # Fallback per PIL più vecchi
                text_width, text_height = draw.textsize(text, font=font)

            x = (target_shape[1] - text_width) / 2
            y = (target_shape[0] - text_height) / 2

            # Disegna il testo in bianco
            draw.text((x, y), text, font=font, fill=255)

            # Converte l'immagine PIL in un array NumPy
            return np.array(img_pil)
        except Exception as e:
            print(f"Errore durante la creazione del watermark di testo: {e}")
            return None

    def _prepare_logo_watermark(self, logo_path, target_shape):
        """
        Carica un logo, lo converte in scala di grigi e lo ridimensiona
        alla forma del sub-band DWT di destinazione.
        """
        try:
            wm = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
            if wm is None:
                raise FileNotFoundError(f"File logo non trovato in: {logo_path}")

            # Ridimensiona il watermark per adattarlo alla banda HH
            wm_resized = cv2.resize(wm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

            # Binarizza il watermark per un segnale più chiaro
            _, wm_binary = cv2.threshold(wm_resized, 127, 255, cv2.THRESH_BINARY)
            return wm_binary
        except Exception as e:
            print(f"Errore durante la preparazione del logo: {e}")
            return None
