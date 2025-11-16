import os

import cv2
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from classes.watermark_mode import IMAGE_MODE, TEXT_MODE

from classes import channels, show_mode
from image_utils.domain_watermark import DomainWatermark


class FrequenceDomainWatermark():


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


    def show_watermark(self,transfomed_img_path):
        original_image = Image.open(self.input_image)
        watermarked_image = Image.open(os.path.join(self.output_dir, "watermarked.png"))
        wm_image = Image.open(os.path.join(self.output_dir, "watermark.png"))
        transformed_img = Image.open(transfomed_img_path)
        self._extract_watermark_dwt(transfomed_img_path, os.path.join(self.output_dir, "transformed_watermark1.png"))

        trans_image = Image.open(os.path.join(self.output_dir, "transformed_watermark1.png"))

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
        plt.imshow(wm_image)
        plt.axis("off")

        if transformed_img:
            plt.subplot(1, 4, 4)
            plt.title("Trasformata")
            plt.imshow(trans_image)
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

        return np.array(text_img)


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




# ... (il resto della tua classe/codice) ...

    def _extract_watermark_dwt_changed(self, watermarked_image_path,
                               output_watermark_path, alpha=0.1, wavelet='haar'):
        """
        Estrae un watermark da un'immagine (metodo NON-BLIND).

        Aggiunge un controllo per ri-allineare le immagini trasformate.

        :param watermarked_image_path: Path dell'immagine con watermark (può essere trasformata).
        :param output_watermark_path: Path dove salvare il watermark estratto.
        :param alpha: La STESSA forza usata durante l'incorporazione.
        :param wavelet: La STESSA wavelet usata durante l'incorporazione.
        """

        # 1. Carica entrambe le immagini
        # Assumiamo che 'self.input_image' sia il path dell'originale
        img_orig = cv2.imread(self.input_image, cv2.IMREAD_COLOR)
        img_wm = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)

        if img_orig is None:
            print(f"Errore: impossibile caricare l'immagine originale da: {self.input_image}")
            return
        if img_wm is None:
            print(f"Errore: impossibile caricare l'immagine con watermark da: {watermarked_image_path}")
            return

        # --- INSERIMENTO CHIAVE: ALLINEAMENTO DIMENSIONI ---

        # Ottieni le dimensioni dell'originale (altezza, larghezza)
        orig_height, orig_width = img_orig.shape[:2]

        # Ottieni le dimensioni dell'immagine con watermark (potrebbe essere trasformata)
        wm_height, wm_width = img_wm.shape[:2]

        # Controlla se le dimensioni sono diverse
        if (orig_height, orig_width) != (wm_height, wm_width):
            print(f"Rilevata discrepanza dimensioni: Originale=({orig_width}x{orig_height}), "
                  f"Trasformata=({wm_width}x{wm_height}).")
            print("Forzo il ridimensionamento dell'immagine trasformata per allinearla...")

            # Ridimensiona forzatamente l'immagine con watermark (img_wm)
            # per farla corrispondere alle dimensioni dell'originale.
            # Questo "inverte" trasformazioni come resize, crop o rotazione.
            # Usiamo INTER_LANCZOS4 per la migliore qualità di resampling (simile a PIL.LANCZOS)
            img_wm = cv2.resize(img_wm, (orig_width, orig_height),
                                interpolation=cv2.INTER_LANCZOS4)

            print(f"Dimensioni dopo il ri-allineamento: {img_wm.shape[:2]}")

        # --- FINE INSERIMENTO ---

        # 2. Converte entrambe in YCrCb e ottieni i canali Y
        # Ora siamo sicuri che img_orig e img_wm abbiano le stesse dimensioni
        Y_orig, _, _ = cv2.split(cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb))
        Y_wm, _, _ = cv2.split(cv2.cvtColor(img_wm, cv2.COLOR_BGR2YCrCb))

        # 3. Applica DWT a entrambi i canali Y
        # Ora le bande HH avranno le stesse dimensioni, risolvendo il ValueError
        coeffs_Y_orig = pywt.dwt2(Y_orig, wavelet)
        _, (_, _, HH_orig) = coeffs_Y_orig

        coeffs_Y_wm = pywt.dwt2(Y_wm, wavelet)
        _, (_, _, HH_wm) = coeffs_Y_wm

        # 4. Estrai il watermark invertendo la formula
        # Formula: watermark = (HH_watermarked - HH) / alpha
        watermark_extracted_raw = (HH_wm - HH_orig) / alpha

        # 5. Post-processa il watermark estratto per renderlo visibile

        # Normalizza l'immagine estratta per visualizzarla (0-255)
        # Usiamo np.nan_to_num per gestire eventuali divisioni per zero se alpha è piccolo
        watermark_extracted_raw = np.nan_to_num(watermark_extracted_raw)
        cv2.normalize(watermark_extracted_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        watermark_visible = watermark_extracted_raw.astype('uint8')

        # Binarizza per pulire il rumore
        _, watermark_final = cv2.threshold(watermark_visible, 127, 255, cv2.THRESH_BINARY)

        # 6. Salva il watermark estratto
        cv2.imwrite(output_watermark_path, watermark_final)
        print(f"Watermark estratto e salvato in: {output_watermark_path}")

        return watermark_final