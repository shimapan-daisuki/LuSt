LuSt
======

Simple yet effective steganography tool written in python2.7 for creating lossless formats images with embedded hidden files.

Overview
======

LuSt is a commandline steganography tool that uses Luma LSB encoding along with key based hash derived insertion.

It can create images with embeded hidden data in lossless formats such as .png, .bmp, .tiff from any source image in Pillow [supported format](https://pillow.readthedocs.io/en/3.3.x/handbook/image-file-formats.html) and any arbitrary file.

Restrictions:
  - installed python2.7 with PIL/Pillow,
  - due to used method of encoding the size of the secret is restricted to 1byte per 8 source image pixels - that translates to 259.2kb for a FHD(1920x1080) image,
  - for the same reason any further conversion of encoded image to lossy formats or certain image manipulations will destroy hidden data.

Features:
  - any file can be encoded as long its size doesn't exceed carrier limit,
  - simple and free using only Pillow and standard library,
  - practically near impossible to detect or decode (tested with [StegExpose](https://github.com/b3dk7/StegExpose)),
  - using multiprocessing for faster encoding/decoding,
  - documented code.


Usage
======

To display help message with list of accepted arguments run:

    python LuSt.py -h

To decode hidden data from example image:

    python LuSt.py --mode decode --input test_images/test+secret_img.png.png --secret secret_img.png -p shimapan.daisuki

Author
======

Found this useful/helpful? You can donate here:

shimapan.daisuki:
- BTC:  15sPxM4hHEghrsZ8egdfubJ6hVesNuxAwd

Contact: shimapan.daisuki@gmail.com
