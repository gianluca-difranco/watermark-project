from image_utils.space_test import LSBWatermarker

from PIL import Image
# Importa le funzioni che abbiamo appena scritto (assumendo che il file si chiami trasformazioni.py)
from image_utils.transformation import resize_image, rotate_image, crop_image, flip_image, adjust_brightness, apply_blur

if __name__ == '__main__':
    # domain_watermark = SpaceDomainWatermark(input_image="files/input.png")
    # domain_watermark.apply_watermark(input_text="HelloWorld!")
    # domain_watermark.show_watermark()
    marker = LSBWatermarker("files")
    marker.embed_watermark("files/input.png", "CIAO ", "files/watermarked_output.png")



    # --- 1. Carica l'immagine originale ---
    img_originale = Image.open("files/watermarked_output.png")
    print(f"Immagine originale caricata: {img_originale.format} {img_originale.size} {img_originale.mode}")

    # --- 2. Applica una catena di trasformazioni ---

    # Nota: applichiamo le trasformazioni in sequenza,
    # passando il risultato della funzione precedente a quella successiva.

    print("Applico trasformazioni...")

    # a. Ridimensiona a una larghezza di 500px mantenendo le proporzioni
    img_temp_1 = resize_image(img_originale, new_width=200, keep_aspect_ratio=True)
    print(f"Dimensione dopo resize: {img_temp_1.size}")

    # b. Ruota l'immagine di 15 gradi (espandendo per non tagliare i bordi)
    # img_temp_2 = rotate_image(img_temp_1, angle=15, fill_color=(0, 0, 0))  # Riempi di nero
    #
    # # c. Ritaglia una regione centrale
    # # Per centrare, calcoliamo il box
    # w, h = img_temp_2.size
    # box = (w / 4, h / 4, w * 3 / 4, h * 3 / 4)  # Prendi il 50% centrale
    # img_temp_3 = crop_image(img_temp_2, box)
    # print(f"Dimensione dopo crop: {img_temp_3.size}")
    #
    # # d. Specchia l'immagine orizzontalmente
    # img_temp_4 = flip_image(img_temp_3, mode='horizontal')
    #
    # # e. Aumenta la luminosità del 20%
    # img_finale = adjust_brightness(img_temp_4, factor=1.2)
    #
    # # --- 3. Salva e mostra il risultato ---
    output_path = "files/watermarked_transformed_output.png"
    img_temp_1.save(output_path)
    # print(f"Immagine trasformata salvata in: {output_path}")

    # Apre l'immagine nel visualizzatore di default del tuo OS
    #img_finale.show()
    marker.show_watermark(transfomed_img_path=output_path)