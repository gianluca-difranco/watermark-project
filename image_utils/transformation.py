from PIL import Image, ImageEnhance, ImageFilter
import os

"""
Una raccolta di funzioni di base per la trasformazione di immagini
utilizzando la libreria Pillow (PIL).

Tutte le funzioni prendono un oggetto Immagine PIL come input
e restituiscono un nuovo oggetto Immagine PIL trasformato.
"""


def resize_image(img, new_width=None, new_height=None, keep_aspect_ratio=True):
    """
    Ridimensiona un'immagine.

    :param img: Oggetto Immagine PIL.
    :param new_width: Nuova larghezza desiderata.
    :param new_height: Nuova altezza desiderata.
    :param keep_aspect_ratio: Se True, mantiene le proporzioni.
                              Richiede che sia new_width O new_height.
    :return: Oggetto Immagine PIL ridimensionato.
    """
    if keep_aspect_ratio:
        if new_width is not None:
            ratio = new_width / float(img.size[0])
            new_height = int(float(img.size[1]) * ratio)
        elif new_height is not None:
            ratio = new_height / float(img.size[1])
            new_width = int(float(img.size[0]) * ratio)
        else:
            # Se non viene fornito né larghezza né altezza, non fare nulla
            return img

    # Usa LANCZOS per un resampling di alta qualità
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def rotate_image(img, angle=30, expand=True, fill_color=(255, 255, 255)):
    """
    Ruota un'immagine di un dato angolo.

    :param img: Oggetto Immagine PIL.
    :param angle: Angolo di rotazione in gradi (senso antiorario).
    :param expand: Se True, l'immagine di output viene ingrandita
                   per contenere l'intera immagine ruotata.
    :param fill_color: Colore per riempire le aree scoperte.
    :return: Oggetto Immagine PIL ruotato.
    """
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=fill_color)


def crop_image(img, box_coords):
    """
    Ritaglia un'immagine in base a una "scatola" di coordinate.

    :param img: Oggetto Immagine PIL.
    :param box_coords: Una tupla di 4 elementi (sinistra, sopra, destra, sotto).
                       (left, upper, right, lower)
    :return: Oggetto Immagine PIL ritagliato.
    """
    return img.crop(box_coords)


def flip_image(img, mode='horizontal'):
    """
    Specchia un'immagine orizzontalmente o verticalmente.

    :param img: Oggetto Immagine PIL.
    :param mode: 'horizontal' (FLIP_LEFT_RIGHT) o 'vertical' (FLIP_TOP_BOTTOM).
    :return: Oggetto Immagine PIL specchiato.
    """
    if mode == 'horizontal':
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img


def adjust_brightness(img, factor):
    """
    Regola la luminosità dell'immagine.

    :param img: Oggetto Immagine PIL.
    :param factor: 1.0 = luminosità originale
                   < 1.0 = più scuro
                   > 1.0 = più luminoso
    :return: Oggetto Immagine PIL con luminosità regolata.
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_blur(img, radius=2):
    """
    Applica una sfocatura gaussiana.

    :param img: Oggetto Immagine PIL.
    :param radius: Raggio della sfocatura.
    :return: Oggetto Immagine PIL sfocato.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))