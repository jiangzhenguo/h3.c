#ifndef H3_PNG_H
#define H3_PNG_H

#include <stddef.h>
#include <stdint.h>

/* Write one interleaved RGB24 image as a PNG (8-bit, non-interlaced).
 * Row 0 is the top row and bytes are ordered R,G,B. */
int h3_png_write_rgb8(const char *path, const uint8_t *rgb, int width,
                      int height, char *error, size_t error_size);

#endif
