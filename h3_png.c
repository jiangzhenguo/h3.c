#include "h3_png.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <zlib.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int make_parents(const char *path, char *error, size_t error_size) {
    char *copy = strdup(path);
    if (!copy) {
        fail(error, error_size, "out of memory resolving PNG directory");
        return 0;
    }
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
            fail(error, error_size, "cannot create PNG directory %s: %s",
                 copy, strerror(errno));
            free(copy);
            return 0;
        }
        *cursor = '/';
    }
    free(copy);
    return 1;
}

static void put_u32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t)(value >> 24u);
    destination[1] = (uint8_t)(value >> 16u);
    destination[2] = (uint8_t)(value >> 8u);
    destination[3] = (uint8_t)value;
}

static int write_chunk(FILE *file, const char type[4], const uint8_t *data,
                       size_t length, char *error, size_t error_size) {
    if (length > UINT32_MAX) {
        fail(error, error_size, "PNG chunk is too large");
        return 0;
    }
    uint8_t header[8];
    put_u32(header, (uint32_t)length);
    memcpy(header + 4, type, 4);
    if (fwrite(header, 1, sizeof(header), file) != sizeof(header)) {
        fail(error, error_size, "cannot write PNG chunk header: %s",
             strerror(errno));
        return 0;
    }
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, (const Bytef *)type, 4);
    if (length) {
        if (fwrite(data, 1, length, file) != length) {
            fail(error, error_size, "cannot write PNG chunk data: %s",
                 strerror(errno));
            return 0;
        }
        crc = crc32(crc, data, (uInt)length);
    }
    uint8_t trailer[4];
    put_u32(trailer, (uint32_t)crc);
    if (fwrite(trailer, 1, sizeof(trailer), file) != sizeof(trailer)) {
        fail(error, error_size, "cannot write PNG chunk checksum: %s",
             strerror(errno));
        return 0;
    }
    return 1;
}

int h3_png_write_rgb8(const char *path, const uint8_t *rgb, int width,
                      int height, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !rgb || width < 1 || height < 1) {
        fail(error, error_size, "invalid PNG output arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / 3 / (size_t)height) {
        fail(error, error_size, "PNG image size overflows");
        return 0;
    }
    size_t stride = (size_t)width * 3;
    /* Every scanline carries one leading filter byte; use filter 2 (Up) for
     * all but the first row, which keeps photographic gradients compact. */
    size_t raw_size = (stride + 1) * (size_t)height;
    uint8_t *raw = malloc(raw_size);
    if (!raw) {
        fail(error, error_size, "out of memory packing PNG scanlines");
        return 0;
    }
    for (int y = 0; y < height; y++) {
        uint8_t *row = raw + (size_t)y * (stride + 1);
        const uint8_t *source = rgb + (size_t)y * stride;
        if (y == 0) {
            row[0] = 0;
            memcpy(row + 1, source, stride);
        } else {
            const uint8_t *above = rgb + (size_t)(y - 1) * stride;
            row[0] = 2;
            for (size_t index = 0; index < stride; index++)
                row[1 + index] = (uint8_t)(source[index] - above[index]);
        }
    }
    uLongf compressed_size = compressBound((uLong)raw_size);
    uint8_t *compressed = malloc(compressed_size);
    if (!compressed) {
        free(raw);
        fail(error, error_size, "out of memory compressing PNG");
        return 0;
    }
    int status = compress2(compressed, &compressed_size, raw, (uLong)raw_size,
                           Z_DEFAULT_COMPRESSION);
    free(raw);
    if (status != Z_OK) {
        free(compressed);
        fail(error, error_size, "cannot compress PNG scanlines (zlib %d)",
             status);
        return 0;
    }
    if (!make_parents(path, error, error_size)) {
        free(compressed);
        return 0;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        free(compressed);
        fail(error, error_size, "cannot create PNG %s: %s", path,
             strerror(errno));
        return 0;
    }
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n',
                                         0x1a, '\n'};
    int ok = fwrite(signature, 1, sizeof(signature), file) == sizeof(signature);
    if (!ok) fail(error, error_size, "cannot write PNG signature: %s",
                  strerror(errno));
    uint8_t ihdr[13];
    put_u32(ihdr, (uint32_t)width);
    put_u32(ihdr + 4, (uint32_t)height);
    ihdr[8] = 8;  /* bit depth */
    ihdr[9] = 2;  /* colour type: truecolour */
    ihdr[10] = 0; /* deflate */
    ihdr[11] = 0; /* adaptive filtering */
    ihdr[12] = 0; /* no interlace */
    if (ok) ok = write_chunk(file, "IHDR", ihdr, sizeof(ihdr), error,
                             error_size);
    if (ok) ok = write_chunk(file, "IDAT", compressed, (size_t)compressed_size,
                             error, error_size);
    if (ok) ok = write_chunk(file, "IEND", NULL, 0, error, error_size);
    free(compressed);
    if (fclose(file) != 0 && ok) {
        ok = 0;
        fail(error, error_size, "cannot finish PNG %s: %s", path,
             strerror(errno));
    }
    return ok;
}
