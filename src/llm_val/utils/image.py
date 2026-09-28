import io
from PIL import Image


def plt2PIL(fig):
    fig.canvas.draw()
    pil_image = Image.frombytes(
        'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return pil_image


def PIL2IOBytes(pil_image, format='png'):
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf
