import argparse
import hashlib
import logging
import multiprocessing
import random
from functools import partial
from itertools import izip

from PIL import Image

from console_printer import *

_HEADER_VALIDATION_BYTES = 3
_HEADER_LEN_BYTES = 3
_MAX_SECRET_BYTE_LEN = 2 ** (_HEADER_LEN_BYTES  * 8)
"""Three integer constants related to header format.

Header consists of 3 validation bytes that correspond to first, second and last byte of password and of 3 byte integer
value representing size of encoded secret in bytes. This allows for maximum of 8388608 bit long secrets(circa 1mb) which
is perfect for used luma single-LSB encoding that requires 1 pixel for one secret bit - UHD-1 (4k) image has just
under 8.3 Mpix.
"""

logging.basicConfig(filename='%s.log' % __file__, level=logging.ERROR, format='%(asctime)s %(message)s')

# todo: add missing comments
# todo: reorder some functions
# todo: implement better header


class IndicesRepeater(object):
    """Generator yielding unique sequence of image indices.

    """

    def __init__(self, pixel_count, seed, hash_func=hashlib.sha256):
        """Init new IndicesRepeater object.

        Args:
            pixel_count: number of pixels
            seed:
            hash_func: hash function used to derive indices
        """
        self.previous = None
        self.highest = 0
        self.pixel_count = pixel_count
        self.hash_seed = seed
        self.hash_func=hash_func
        self.injected_indices = set()
        self.stuck_threshold = self.pixel_count / (256*256)

    def repeatable_random_sequence(self):
        """Generates sequence of values based on seeded hash function and previous value.

        Yields:
            int: 'random' value
        """
        while True:
            self.hash_seed = self.hash_func(self.hash_seed).digest()
            digest_iter = iter(self.hash_seed)
            if self.previous is None:
                self.previous = ord(digest_iter.next())
            for c in digest_iter:
                c_val = ord(c)
                nxt = c_val * self.previous
                self.previous = c_val
                yield nxt

    def next_free_index(self, index):
        """Checks if index has been used already..
        If so then it generates/finds not yet used indice.

        Args:
            index: integer

        Returns:
            integer: not used indice
        """
        next_index = index
        stuck = 0
        used_count = len(self.injected_indices)
        # speedup for last 10% of indices
        end_fill = float(used_count) / float(self.pixel_count) > 0.9
        while next_index in self.injected_indices or next_index >= self.pixel_count or next_index < 0:
            if not end_fill:
                if stuck > self.stuck_threshold:
                    v = self.repeatable_random_sequence().next()
                    next_index += (self.stuck_threshold * v)
                    stuck = 0
                if next_index >= self.pixel_count:
                    next_index -= self.pixel_count
                else:
                    next_index += used_count
                stuck += 1
            else:
                for n in xrange(self.highest, self.pixel_count+1):
                    if n not in self.injected_indices:
                        self.highest = next_index = n
                        break
        return next_index

    def _next(self):
        """Yields indices.
        """
        if len(self.injected_indices) == self.pixel_count:
            raise StopIteration
        for v in self.repeatable_random_sequence():
            indice = self.next_free_index((self.pixel_count / (256 * 256) * v) + v)
            self.injected_indices.add(indice)
            yield indice

    def __iter__(self):
        return self

    def next(self):
        return self._next().next()


def _ycc(r, g, b, a=None):
    """Conversion from rgba to ycc.

    Notes:
         Expects and returns values in [0,255] range.
    """
    y = .299*r + .587*g + .114*b
    cb = 128 - .168736*r - .331364*g + .5*b
    cr = 128 + .5*r - .418688*g - .081312*b
    if a is not None:
        return [y, cb, cr, a]
    else:
        return [y, cb, cr]


def _rgb(y, cb, cr, a=None):
    """Conversion from ycc to rgba.

     Notes:
         Expects and returns values in [0,255] range.
    """
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) - .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    if a is not None:
        return [int(round(r)), int(round(g)), int(round(b)), int(a)]
    else:
        return [int(round(r)), int(round(g)), int(round(b))]


def set_last_bit(value, target):
    """Sets target as LSB in given value.

    Args:
        value: value to encode as LSB.
        target: 0 or 1 - bit value to encode.

    Returns:
        int: encoded value
    """
    if target:
        out = value | target
    else:
        out = value >> 1 << 1
    return out


def encode_pixel(pixel, bit):
    """Encodes given bit in given pixel luma value.

    Args:
        pixel: tuple representing pixel's rgb/rgba values
        bit: bit to encode

    Returns:
        tuple: tuple representing encoded pixel's rgb/rgba values
    """
    pix_ycc = _ycc(*pixel)
    sec = pix_ycc[0]
    sec = int(round(sec))
    injected = set_last_bit(sec, bit)
    pix_sec = _rgb(injected, *pix_ycc[1:])
    if 256 in pix_sec:
        pix_sec = _rgb(injected - 2, *pix_ycc[1:])
    if -1 in pix_sec:
        pix_sec = _rgb(injected + 2, *pix_ycc[1:])
    return pix_sec


def decode_pixel(pixel):
    """Returns luma LSB from given rgb/a pixel values.
    """
    pix_ycc = _ycc(*pixel)
    sec = pix_ycc[0]
    sec = int(round(sec))
    return sec & 0b1


def value_from_list_of_bits(lst):
    """Converts given list of bits to int value.
    """
    val = str()
    for bit in lst:
        val += str(bit)
    val = int(val, 2)
    return val


def list_of_bits_from_value(value, fill_zeroes_to=8):
    """Creates list of bits from given int value.

    Args:
        value: integer value
        fill_zeroes_to: zero padding from left

    Returns:
        list: binary representation of given value
    """
    val = [((value >> n) & 1) for n in reversed(xrange(len(bin(value))-2))]
    val = [0]*(fill_zeroes_to-len(val)) + val
    return val


def secret_to_bits(secret):
    """Generates binary representation of given str/list of bytes.

    Notes:
        Bits are yielded in order from MSB to LSB.

    Args:
        secret: str or list of (8 bit long byte) values

    Yields:
        int: next bit value in binary representation of secret
    """
    for byte in secret:
        bits_list = list_of_bits_from_value(ord(byte))
        for bit in bits_list:
            yield bit


def indice_to_pos(image_size, indice):
    """Converts indice of a pixel in image(index number in array of pixels) to x,y coordinates.

    Args:
        image_size:
        indice: index number in image's array of pixels.

    Returns:
        tuple: pair of values representing x,y coordinates corresponding to given indice.
    """
    size_x, size_y = image_size
    pos_y = indice / size_x
    pos_x = indice % size_x
    pos = pos_x, pos_y
    return pos


def assert_sizes(pixel_count, secret_bit_len, encode_bit_pix_density=1):
    """Validate secret and image sizes.

    Notes:
        Secret can't be longer than maximum length supported by header format.
        Size of secret and header cannot exceed constrains set by encoding and image size.
    """
    assert secret_bit_len <= _MAX_SECRET_BYTE_LEN * 8
    assert secret_bit_len + (_HEADER_LEN_BYTES+_HEADER_VALIDATION_BYTES)*8 <= pixel_count * encode_bit_pix_density


def create_header_bits(secret_byte_len, password):
    """Creates header containing validation bytes and secret length.

    Args:
        secret_byte_len: size of secret in bytes.
        password: seed for generating indices sequence, first,second and last char are used for validation.

    Returns:
        list: binary representation of header.
    """
    a, b = list(password[:2])
    password_letters = [a, b, (password[-1])]
    validation_bits = list()
    for char in password_letters:
        validation_bits.extend(list_of_bits_from_value(ord(char)))
    secret_len_as_bits = list_of_bits_from_value(secret_byte_len, fill_zeroes_to=_HEADER_LEN_BYTES*8)

    header_bits = validation_bits + secret_len_as_bits
    assert len(header_bits) == (_HEADER_LEN_BYTES+_HEADER_VALIDATION_BYTES)*8
    return header_bits


def validate_header(header_bits, password):
    """Validates sequence of bits as valid header for given password and if so then returns secret length.

    Args:
        header_bits: sequence of bits to be validated.
        password: seed for generating indices sequence, first,second and last char are used for validation.

    Returns:
        int: length secret in bytes.
    """
    a, b = list(password[:2])
    password_letters = [a, b, (password[-1])]
    validation_bits_from_pass = list()
    for char in password_letters:
        validation_bits_from_pass.extend(list_of_bits_from_value(ord(char)))
    validation_bits_from_header = header_bits[:_HEADER_VALIDATION_BYTES*8]

    assert validation_bits_from_header == validation_bits_from_pass
    secret_len_bits = header_bits[_HEADER_VALIDATION_BYTES*8:]
    return value_from_list_of_bits(secret_len_bits)


def calculate_bit_distribution(source, input_as_list_of_bits=False):
    """Calculates distribution of zeroes and ones in given list of bytes or bits.

    Args:
        source: list of bits/bytes/single byte values.
        input_as_list_of_bits: set True if source is list of bit values.

    Returns:
        tuple: pair of values representing count of zeroes and count of ones in source binary representation.
    """
    one_count, zero_count = 0, 0
    secret_bits = source
    if input_as_list_of_bits:
        source = [True]
    for b in source:
        if input_as_list_of_bits:
            bits_list = secret_bits
        else:
            bits_list = list_of_bits_from_value(ord(b))
        for bit in bits_list:
            if bit:
                one_count += 1
            else:
                zero_count += 1
    return zero_count, one_count


def supply_pixels(indices_iter, bits_iter, image_pixels, image_size, debug_pix=None):
    """Generator for encoding/decoding using multiprocessing.Pool.

    Args:
        indices_iter: unique sequence of image's pixel indices to which encode secret bits.
        bits_iter: secret as sequence of binary values to encode.
        image_pixels: PIL.Image.load() object allowing for fast get/set pixel operations.
        image_size: tuple with images dimensions - pair of length and height values.
        debug_pix: optional pixel as tuple of rgba values to use during encoding instead of actual image pixels.

    Yields:
        tuple: triplet of pixel color values as rgba tuple, bit to be encoded in it and tuple of pixel's coordinates.
    """
    for indice, bit in izip(indices_iter, bits_iter):
        pos = indice_to_pos(image_size, indice)
        if debug_pix:
            pxl = debug_pix
        else:
            pxl = image_pixels[pos]
        yield pxl, bit, pos


def encode_func_wrapper(arguments, skip_assert=False):
    """Wrapper function for encoding pixel for use in multiprocessing.Pool.

    Args:
        arguments: triplet of pixel as rgba tuple, bit to be encoded in it and tuple of its coordinates.
        skip_assert: optional set True to omit checking if decoding encoded pixel returns correct value

    Returns:
        tuple: encoded pixel as pair of tuple with coordinates and tuple with rgb/a values.
    """
    pixel, bit, pos = arguments
    encoded_pxl = encode_pixel(pixel, bit)
    if not skip_assert:
        assert bit == decode_pixel(encoded_pxl)
    return pos, encoded_pxl


def encode_bits(image_to_encode, indices_iter, bits_iter, debug_pix=None, printf=None, secret_bit_len=None,
                skip_assert=False):
    """Encode secret bits in given image[s pixels.

    Notes:
        This method modifies passed image object - encoding is done in-place.
        Passing printf requires to pass secret_bin_len as well as it's used for calculating encoding progress.

    Args:
        image_to_encode: PIL.Image object in which we'll encode secret.
        indices_iter: unique sequence of image's pixel indices to which encode secret bits.
        bits_iter: secret as sequence of binary values to encode.
        debug_pix: optional pixel as tuple of rgba values to use during encoding instead of actual image pixels.
        printf: optional ConsolePrinter object to use for printing encoding status.
        secret_bit_len: optional length of secret in bits.
        skip_assert: optional set True to omit checking if decoding encoded pixel returns correct value
    """
    count = 0
    last = 0
    image_pixels = image_to_encode.load()
    for n, bit in izip(indices_iter, bits_iter):
        pos = indice_to_pos(image_to_encode.size, n)
        if debug_pix:
            new_pxl = encode_pixel(debug_pix, bit)
        else:
            new_pxl = encode_pixel(image_pixels[pos], bit)
        if not skip_assert:
            assert bit == decode_pixel(new_pxl)

        image_pixels[pos] = tuple(new_pxl)
        if printf:
            count += 1
            status = count * 100.0 / secret_bit_len
            if status - last >= 0.8:
                last = status
                printf.display(progress='[%3.2f%%]' % status)


def encode_bits_mp(image_to_encode, indices_iter, bits_iter, debug_pix=None, printf=None, secret_bit_len=None,
                   process_count=2, skip_assert=False):
    """Encode secret bits in given image's pixels using multiprocessing.Pool.

    Notes:
        This method modifies passed image object - encoding is done in-place.
        Passing printf requires to pass secret_bin_len as well as it's used for calculating encoding progress.

    Args:
        image_to_encode: PIL.Image object in which we'll encode secret.
        indices_iter: unique sequence of image's pixel indices to which encode secret bits.
        bits_iter: secret as sequence of binary values to encode.
        debug_pix: optional pixel as tuple of rgba values to use during encoding instead of actual image pixels.
        printf: optional ConsolePrinter object to use for printing encoding status.
        secret_bit_len: optional length of secret in bits.
        process_count: number of worker processes to use.
        skip_assert: optional set True to omit checking if decoding encoded pixel returns correct value
    """
    count = 0
    last = 0
    image_pixels = image_to_encode.load()
    pool = multiprocessing.Pool(process_count)
    supply = supply_pixels(indices_iter, bits_iter, image_pixels, image_to_encode.size, debug_pix)
    if skip_assert:
        func = partial(encode_func_wrapper, skip_assert=True)
    else:
        func = encode_func_wrapper

    chunk_len = 1280 / process_count
    if debug_pix:
        _ = time.time()
    for pos, encoded_pixel in pool.imap_unordered(func, supply, chunksize=chunk_len):
        image_pixels[pos] = tuple(encoded_pixel)
        if printf:
            count += 1
            status = count * 100.0 / secret_bit_len
            if status - last >= 0.8:
                last = status
                printf.display(progress='[%3.2f%%]' % status)
    if debug_pix:
        print 'encode time: ', time.time()-_


def decode_func_wrapper(arguments):
    """Wrapper function for decoding pixel for use in multiprocessing.Pool.

    Args:
        arguments: triplet of pixel as rgb/a tuple, bit to be encoded in it and tuple of its coordinates.

    Returns:
        int: decoded binary value
    """
    pixel, bit, pos = arguments
    return decode_pixel(pixel)


def decode_bits_mp(image_to_decode, indices_iter, secret_bit_len, printf=None, process_count=2):
    """Decode secret bits in given image's pixels using multiprocessing.Pool,

    Args:
        image_to_decode: PIL.Image object from which we'll decode the secret.
        indices_iter: unique sequence of image's pixel indices from which encode secret bits.
        secret_bit_len: length of encoded secret in bits.
        printf: optional ConsolePrinter object to use for printing encoding status.
        process_count: number of worker processes to use.

    Returns:
        list: sequence of binary representing decoded secret bits.
    """
    count = 0
    last = 0
    image_pixels = image_to_decode.load()
    pool = multiprocessing.Pool(process_count)
    supply = supply_pixels(indices_iter, xrange(secret_bit_len), image_pixels, image_to_decode.size)
    decoded_bits = list()
    chunk_len = 640 / process_count
    for bit in pool.imap(decode_func_wrapper, supply, chunksize=chunk_len):
        decoded_bits.append(bit)
        if printf:
            count += 1
            status = count * 100.0 / secret_bit_len
            if status - last >= 0.8:
                last = status
                printf.display(progress='[%3.2f%%]' % status)
    return decoded_bits


def decode_bits(image_to_decode, indices_iter, secret_bit_len, printf=None):
    """Decode secret bits in given image's pixels.

    Args:
        image_to_decode: PIL.Image object from which we'll decode the secret.
        indices_iter: unique sequence of image's pixel indices from which encode secret bits.
        secret_bit_len: length of encoded secret in bits.
        printf: optional ConsolePrinter object to use for printing encoding status.

    Returns:
        list: sequence of binary representing decoded secret bits,
    """
    count = 0
    last = 0
    image_pixels = image_to_decode.load()

    decoded_bits = list()

    for n, _ in izip(indices_iter, secret_bit_len):
        pos = indice_to_pos(image_to_decode.size, n)
        bit = decode_pixel(image_pixels[pos])
        decoded_bits.append(bit)

        if printf:
            count += 1
            status = count * 100.0 / secret_bit_len
            if status - last >= 0.8:
                last = status
                printf.display(progress='[%3.2f%%]' % status)
    return decoded_bits


def noisify_image(image_to_noisify, secret, indices_used, original_distribution, printf=None, process_count=None):
    """Encode all pixels not containing secret with noise with its distribution matching source image.

    Notes:
        This method modifies passed image object - encoding is done in-place.

    Args:
        image_to_noisify: PIL.Image object to noisify,.
        secret: secret as str or list of bytes/char values.
        indices_used: set containing indexes of pixels encoded with secret
        original_distribution: distribution of zeroes and ones in luma LSB in source image.
        printf: optional ConsolePrinter object to use for printing encoding status.
        process_count: optional number of worker processes to use.
    """
    secret_dist = calculate_bit_distribution(secret)
    leftover_dist = (original_distribution[0] - secret_dist[0], original_distribution[1] - secret_dist[1])

    pixel_count = image_to_noisify.size[0] * image_to_noisify.size[1]
    zero_chance = float(leftover_dist[0])/float(leftover_dist[0]+leftover_dist[1])

    noise_bits = (0 if random.random() < zero_chance else 1 for _ in xrange(pixel_count-len(indices_used)))
    indices_to_noise = (n for n in xrange(pixel_count) if n not in indices_used)

    if process_count == 0:
        encode_bits(image_to_noisify, indices_to_noise, noise_bits, printf=printf,
                    secret_bit_len=pixel_count-len(indices_used), skip_assert=True)
    else:
        encode_bits_mp(image_to_noisify, indices_to_noise, noise_bits, printf=printf,
                       secret_bit_len=pixel_count - len(indices_used), process_count=process_count, skip_assert=True)


def encode_image(image_to_encode, secret, password, output_filename, verbose=True, debug=False, noisify=True,
                 process_count=None, skip_assert=False):
    """Encode secret inside given image.

    Args:
        image_to_encode: PIL.Image object with source image to use for encoding secret,
        secret: secret as str or list of bytes/char values.
        password: seed for generating indices sequence
        output_filename: full path with name for encoded image.
        verbose: optional set False to disable console messages.
        debug: optional set True to run in debug mode
        noisify: optional set True to noisify image after encoding secret.
        process_count: optional number of worker processes to use.
        skip_assert: optional set True to omit checking during encode if decoding encoded pixel returns correct value

    Returns:
        list: optional list containing additional information from running in debug mode
    """
    header_debug_pix, data_debug_pix, debug_log = None, None, None
    if debug:
        header_debug_pix, data_debug_pix, debug_log = [255, 0, 255], [0, 255, 0], list()

    printf = ConsolePrinter("{phase} {progress}\n", _clear=False, phase='', progress='', _disabled=not verbose)

    if process_count == 0:
        process_count = multiprocessing.cpu_count()

    if debug:
        printf.display(True, phase='MODE 1+%d:' % process_count, progress='ENCODING IN DEBUG MODE')
    else:
        printf.display(True, phase='MODE 1+%d:' % process_count, progress='ENCODING')
    printf.display(True, phase='init...', progress='')
    if noisify:
        orig_distribution = calculate_bit_distribution((decode_pixel(pxl) for pxl in image_to_encode.getdata()),
                                                       input_as_list_of_bits=True)
    size_x, size_y = image_to_encode.size
    pixel_count = size_x * size_y

    secret_bit_len = len(secret)*8
    try:
        assert_sizes(pixel_count, secret_bit_len)
    except:
        raise OverflowError('Secret is too large to be encoded in given image.')
    indices_to_inject = IndicesRepeater(pixel_count, password)

    printf.display(True, phase='creating header...')
    header = create_header_bits(secret_bit_len/8, password)
    encode_bits(image_to_encode, indices_to_inject, header, header_debug_pix)

    printf.display(True, phase='encoding data...', progress='[0%]')
    data = secret_to_bits(secret)

    if process_count == 0:
        encode_bits(image_to_encode, indices_to_inject, data, data_debug_pix, printf, secret_bit_len, skip_assert)
    else:
        encode_bits_mp(image_to_encode, indices_to_inject, data, data_debug_pix, printf, secret_bit_len, process_count,
                       skip_assert)

    printf.display(True, progress='[100%]')
    if noisify:
        printf.display(True, phase='noisyfing image...', progress='')
        printf.display(True, phase='noisyfing image...', progress='[0%]')
        noisify_image(image_to_encode, data, indices_to_inject.injected_indices, orig_distribution, printf,
                      process_count)
        printf.display(True, phase='noisyfing image...', progress='[100%]')

    if args.debug:
        output_filename = 'debug_n%d_%d.png' % (args.noisify, time.time())
    image_to_encode.save(output_filename)

    printf.display(True, phase='done.', progress='')
    return debug_log


def decode_image(image_to_decode, password, debug=False, verbose=True, process_count=None):
    """Decode secret from given image.

    Args:
        image_to_decode: PIL.Image object with source image containing encoded secret,
        password: seed for generating indices sequence
        verbose: optional set False to disable console messages.
        debug: optional set True to run in debug mode
        process_count: optional number of worker processes to use.

    Returns:
        tuple: returns tuple consisting of list of decoded bytes and list with additional debug information
    """
    debug_log = None
    if debug:
        debug_log = list()

    printf = ConsolePrinter("{phase} {progress}\n", _clear=False, phase='', progress='', _disabled=not verbose)
    if debug:
        printf.display(True, phase='MODE 1+%d:' % process_count, progress='DECODING IN DEBUG MODE')
    else:
        printf.display(True, phase='MODE 1+%d:' % process_count, progress='DECODING')

    size_x, size_y = image_to_decode.size
    pixel_count = size_x * size_y
    indices_to_inject = IndicesRepeater(pixel_count, password)

    printf.display(True, phase='reading header...', progress='')
    try:
        header_bits = decode_bits(image_to_decode, indices_to_inject,
                                  xrange((_HEADER_LEN_BYTES+_HEADER_VALIDATION_BYTES) * 8))

        secret_bit_len = validate_header(header_bits, password)*8
        assert_sizes(pixel_count, secret_bit_len)
    except:
        print("Error while reading the header. Make sure you are using the proper encryption password.")
        raise

    printf.display(True, phase='decoding data...', progress='[0%]')
    if process_count == 0:
        decoded_data_bits = decode_bits(image_to_decode, indices_to_inject, secret_bit_len, printf=printf)
    else:
        decoded_data_bits = decode_bits_mp(image_to_decode, indices_to_inject, secret_bit_len, printf=printf,
                                           process_count=process_count)
    printf.display(True, phase='decoding data...', progress='[100%]')

    printf.display(True, phase='decoding bytes...', progress='')
    decoded_bytes = [decoded_data_bits[i:i + 8] for i in xrange(0, len(decoded_data_bits), 8)]
    secret_bytes = bytearray(value_from_list_of_bits(byte_as_list_of_bits) for byte_as_list_of_bits in decoded_bytes)

    printf.display(True, phase='done.')
    return secret_bytes, debug_log


def benchmark(resolution):
    """Simple benchmark printing time to generate indices, decode and encode for 1/10th of pixles in given resolution.

    Notes:
        Calls sys,exit after finishing.

    Args:
        resolution: image resolution to benchmark for.
    """
    width, height = resolution
    pixel_count = int(width)*int(height)
    indices = IndicesRepeater(pixel_count, str(time.time()))
    print 'bechmarking for %s, number of pixels:' % resolution, pixel_count
    st = time.time()
    for _, __ in izip(indices, xrange(int(pixel_count/10))):
        pass
    ed = time.time()
    indice_gen_time = ed-st
    print 'time to gen indices for 1/10th of pixels', indice_gen_time
    pixels = [[random.randint(0, 255)]*3 for _ in xrange(pixel_count/10)]
    bits = [random.randint(0, 1) for _ in xrange(pixel_count/5)]
    st = time.time()
    for pix, bit in izip(pixels, bits):
        _ = encode_pixel(pix, bit)
    ed = time.time()
    encode_pix_time = ed-st
    print 'time to encode 1/10th of pixels', encode_pix_time
    st = time.time()
    for pix in pixels:
        _ = decode_pixel(pix)
    ed = time.time()
    decode_pix_time = ed-st
    print 'time to decode 1/10th of pixels', decode_pix_time
    sys.exit()


def run_encoding_mode(image_to_encode, settings):
    """Executes encoding secret in image.
    """
    try:
        with open(settings.secret, 'rb') as f:
            secret_data = f.read()
    except Exception as ex:
        print("Couldn't open secret: %s\nDetails: %s" % (settings.secret, ex))
        raise

    try:
        encode_image(image_to_encode, secret_data, password=settings.password, output_filename=settings.output,
                     verbose=settings.verbose, debug=settings.debug, noisify=settings.noisify,
                     process_count=settings.cores)
    except Exception as ex:
        print("Couldn't encode image: %s\nwith secret: %s\nDetails: %s" % (settings.input, settings.secret, ex))
        raise


def run_decoding_mode(image_to_decode, settings):
    """Executes decoding secret from image,
    """
    try:
        decoded_bytes, debug_log = decode_image(image_to_decode, password=settings.password, debug=settings.debug,
                                                verbose=settings.verbose, process_count=settings.cores)
    except Exception as ex:
        print("Couldn't decode image: %s\nDetails: %s" % (settings.input, ex))
        raise

    try:
        with open(settings.secret, 'wb') as f:
            f.write(str(decoded_bytes))
    except Exception as ex:
        print("Couldn't save secret: %s\nDetails: %s" % (settings.secret, ex))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mode", type=str, choices=['d', 'decode', 'e', 'encode'], required=True,
                        help='choose the usage mode from encoding a secret into image and decoding secret from image')
    parser.add_argument('-i', "--input", type=str, default='input.png', help='path to the input image.')
    parser.add_argument('-s', "--secret", type=str, default='secret', help='path for the input/output secret')
    parser.add_argument('-p', "--password", type=str, help='specify encryption password to encode/decode secret,'
                        'if not specified script will try to use --input for decoding and --output for encoding')
    parser.add_argument('-o', "--output", type=str, default='output.png', help='path to the output image')
    parser.add_argument('-d', "--debug", default=False, action="store_true", help='set to run in debug mode')
    parser.add_argument('-v', "--verbose", default=True, action="store_false", help='disable verbose mode')
    parser.add_argument('-n', "--noisify", default=True, action="store_false", help='disable noisifying output image,'
                        'not recommended unless for very small payloads(less than 10% of maximum)')
    parser.add_argument('-c', "--cores", default=2, type=int, help='set additional processes value, default is 2')
    parser.add_argument('-u', "--unsafe", default=False, action="store_true", help='run in unsafe mode')
    parser.add_argument('-b', "--benchmark", default=False, type=str,
                        help='run simple single process benchmark for given resolution, format: 100x100')

    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.benchmark.split('x'))
    start = time.time()
    try:
        try:
            image = Image.open(args.input)
        except Exception as e:
            print("Error.\nCouldn't open input image: %s\nDetails: %s" % (args.input, e))
            raise

        if args.mode in {'e', 'encode'}:
            if args.password is None:
                args.password = args.output
            try:
                run_encoding_mode(image, args)
            except:
                print("Error while encoding.")
                raise
        else:
            if args.password is None:
                args.password = args.input
            try:
                run_decoding_mode(image, args)
            except:
                print("Error while decoding.")
                raise
    except Exception:
        logging.exception('')
        sys.exit("Couldn't successfully execute script. Saving traceback to log and exiting.")
    else:
        end = time.time()
        print 'Finished successfully after %d seconds.' % (end-start)
