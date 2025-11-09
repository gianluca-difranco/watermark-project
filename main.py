from image_utils.utils import apply_watermark
from image_utils.space_domain_watermark import SpaceDomainWatermark

if __name__ == '__main__':
    domain_watermark = SpaceDomainWatermark(input_image="files/input.png")
    domain_watermark.apply_watermark(input_text="HelloWorld!")
    domain_watermark.show_watermark()