LuSteg
======

Simple yet effective steganography tool written in python2.7 for creating (.PNG,.BMP) images with embedded hidden files.

Usage and basics
======

LuSteg allows to embed hidden 
Restrictions:
  - installed python2.7
  - due to used method of encoding the size of the secret is restricted to 1byte per 8 source image pixels.
  That translates to 259.2kb for a FHD(1920x1080) image.

Any file can be encoded as long its size doesn't exceed that limit.

LuSteg is commandline tool.

To display help message with list of accepted arguments run:

    python LuSteg.py -h

