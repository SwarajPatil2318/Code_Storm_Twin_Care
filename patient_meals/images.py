"""Verify image bytes and produce EXIF-free pixels for the analysis provider."""
import io
import warnings
from PIL import Image,ImageOps

MAX_BYTES=8*1024*1024
FORMATS={'JPEG':'image/jpeg','PNG':'image/png','WEBP':'image/webp'}


def validated_image(raw,mime):
    if not isinstance(raw,bytes) or not 0<len(raw)<=MAX_BYTES:raise ValueError('Choose an image smaller than 8 MB')
    if mime not in FORMATS.values():raise ValueError('JPEG, PNG or WEBP required')
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in FORMATS or FORMATS[im.format]!=mime:raise ValueError('Image MIME mismatch')
            if im.width*im.height>16_000_000 or max(im.size)>8000 or min(im.size)<32:raise ValueError('Image dimensions not supported')
            if getattr(im,'n_frames',1)!=1:raise ValueError('Upload a single still meal photo')
            kind=im.format;size=im.size;im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            pixels=ImageOps.exif_transpose(im).convert('RGB');pixels.thumbnail((1024,1024))
            pixels.info.clear()
            out=io.BytesIO();pixels.save(out,format='PNG')
    return {'format':kind,'width':size[0],'height':size[1],'provider_pixels':out.getvalue(),'provider_mime':'image/png'}
