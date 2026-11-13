#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include "h3_image_vae.h"

#include <stddef.h>
#include <stdint.h>

typedef int (*h3_image_progress)(const char *phase, int completed, int total,
                                 void *opaque);

typedef struct {
    int width;
    int height;
    uint64_t seed;
    int ssd_streaming;
    /* Diagnostic only: fewer than 50 blocks produces an invalid but fast
     * pipeline check. The released model uses 50. */
    int layers;
    /* Directory containing the shared MiniMax-H3 Qwen3-VL text encoder. */
    const char *text_encoder_directory;
    /* Ordered reference images of an edit; the prompt refers to them as
     * <Picture 1>, <Picture 2>, ... The released model accepts up to 16. */
    const char *const *reference_images;
    size_t reference_count;
    h3_image_progress on_progress;
    void *callback_opaque;
} h3_image_params;

#define H3_IMAGE_PARAMS_DEFAULT ((h3_image_params){ \
    .width = 2048, .height = 2048, .seed = 42, .ssd_streaming = 1, \
    .layers = 50, \
    .text_encoder_directory = NULL, .reference_images = NULL, \
    .reference_count = 0, .on_progress = NULL, \
    .callback_opaque = NULL \
})

/* Native MiniMax-H3-Image text-to-image, or reference-image editing when
 * params->reference_count is non-zero. The returned RGB24 buffer is owned by
 * the caller and is released with h3_image_vae_output_free(). */
int h3_image_generate(const char *model_directory,
                      const char *shader_source_path, const char *prompt,
                      const h3_image_params *params,
                      h3_image_vae_output *output,
                      char *error, size_t error_size);

#endif
