#include "h3_png.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

static int tests_run;

#define CHECK(condition)                                                       \
    do {                                                                       \
        tests_run++;                                                           \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static uint32_t read_u32(const uint8_t *source) {
    return ((uint32_t)source[0] << 24u) | ((uint32_t)source[1] << 16u) |
           ((uint32_t)source[2] << 8u) | (uint32_t)source[3];
}

/* Minimal PNG reader for the files this project writes: it walks the chunks,
 * checks every CRC, and returns the decompressed scanlines. */
static uint8_t *decode(const char *path, uint32_t *width, uint32_t *height,
                       size_t *raw_size, size_t *chunk_count) {
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    long size = ftell(file);
    CHECK(size > 0 && fseek(file, 0, SEEK_SET) == 0);
    uint8_t *data = malloc((size_t)size);
    CHECK(data != NULL);
    CHECK(fread(data, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n',
                                         0x1a, '\n'};
    CHECK(memcmp(data, signature, 8) == 0);
    size_t cursor = 8, idat_size = 0;
    uint8_t *idat = NULL;
    int saw_header = 0, saw_end = 0;
    *chunk_count = 0;
    while (cursor + 12 <= (size_t)size) {
        uint32_t length = read_u32(data + cursor);
        const uint8_t *type = data + cursor + 4;
        const uint8_t *payload = data + cursor + 8;
        CHECK(cursor + 12 + (size_t)length <= (size_t)size);
        uLong crc = crc32(0L, Z_NULL, 0);
        crc = crc32(crc, type, 4);
        if (length) crc = crc32(crc, payload, length);
        CHECK((uint32_t)crc == read_u32(payload + length));
        (*chunk_count)++;
        if (!memcmp(type, "IHDR", 4)) {
            CHECK(length == 13);
            *width = read_u32(payload);
            *height = read_u32(payload + 4);
            CHECK(payload[8] == 8 && payload[9] == 2 && payload[12] == 0);
            saw_header = 1;
        } else if (!memcmp(type, "IDAT", 4)) {
            uint8_t *grown = realloc(idat, idat_size + length);
            CHECK(grown != NULL);
            idat = grown;
            memcpy(idat + idat_size, payload, length);
            idat_size += length;
        } else if (!memcmp(type, "IEND", 4)) {
            saw_end = 1;
        }
        cursor += 12 + (size_t)length;
    }
    CHECK(saw_header && saw_end && idat != NULL);
    *raw_size = (size_t)(*width) * 3 + 1;
    *raw_size *= (size_t)*height;
    uint8_t *raw = malloc(*raw_size + 1);
    CHECK(raw != NULL);
    uLongf produced = (uLongf)*raw_size;
    CHECK(uncompress(raw, &produced, idat, (uLong)idat_size) == Z_OK);
    CHECK((size_t)produced == *raw_size);
    free(idat);
    free(data);
    return raw;
}

int main(void) {
    const int width = 7, height = 5;
    size_t stride = (size_t)width * 3;
    uint8_t *rgb = malloc(stride * (size_t)height);
    CHECK(rgb != NULL);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int channel = 0; channel < 3; channel++)
                rgb[((size_t)y * (size_t)width + (size_t)x) * 3 +
                    (size_t)channel] =
                    (uint8_t)(y * 37 + x * 11 + channel * 53);
    char error[256];
    CHECK(h3_png_write_rgb8("/tmp/h3-png-test/nested/out.png", rgb, width,
                            height, error, sizeof(error)));
    uint32_t read_width = 0, read_height = 0;
    size_t raw_size = 0, chunk_count = 0;
    uint8_t *raw = decode("/tmp/h3-png-test/nested/out.png", &read_width,
                          &read_height, &raw_size, &chunk_count);
    CHECK(read_width == (uint32_t)width);
    CHECK(read_height == (uint32_t)height);
    CHECK(chunk_count == 3);
    /* Undo the per-row filters and compare against the source pixels. The Up
     * filter predicts from the previous *reconstructed* row. */
    uint8_t *recon = malloc(stride * (size_t)height);
    CHECK(recon != NULL);
    for (int y = 0; y < height; y++) {
        const uint8_t *row = raw + (size_t)y * (stride + 1);
        uint8_t *out = recon + (size_t)y * stride;
        CHECK(row[0] == (y == 0 ? 0 : 2));
        for (size_t index = 0; index < stride; index++) {
            uint8_t value = row[1 + index];
            if (y > 0)
                value = (uint8_t)(value + recon[(size_t)(y - 1) * stride +
                                                index]);
            out[index] = value;
        }
    }
    CHECK(memcmp(recon, rgb, stride * (size_t)height) == 0);
    free(recon);
    free(raw);
    CHECK(h3_png_write_rgb8("/tmp/h3-png-bad.png", NULL, width, height, error,
                            sizeof(error)) == 0);
    CHECK(h3_png_write_rgb8("/tmp/h3-png-bad.png", rgb, 0, height, error,
                            sizeof(error)) == 0);
    free(rgb);
    printf("ok %d PNG writer tests\n", tests_run);
    return 0;
}
