# -*- coding: utf-8 -*-
# @Time    : 9/5/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : image.py


from io import BytesIO

from PIL import Image

__all__ = ['byte_imread', "byte_imwrite"]


def byte_imread(data):
    return Image.open(BytesIO(data))


def byte_imwrite(image, quality=100, subsampling=0):
    image = Image.fromarray(image)
    with BytesIO() as f:
        image.save(f, format="JPEG", quality=quality, subsampling=subsampling)
        data = f.getvalue()
    return data
