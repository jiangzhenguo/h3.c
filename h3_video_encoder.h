#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* A resident encoder keeps the 2.3 GiB weight set and the Metal pipeline alive
 * across calls. Editing an image with several references encodes each of them
 * through the same encoder, so the load happens once instead of per image. */
typedef struct h3_video_vae_encoder h3_video_vae_encoder;

h3_video_vae_encoder *h3_video_vae_encoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        char *error, size_t error_size);
int h3_video_vae_encoder_encode(h3_video_vae_encoder *encoder,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress,
                        void *progress_opaque, h3_video_latent *output,
                        char *error, size_t error_size);
void h3_video_vae_encoder_free(h3_video_vae_encoder *encoder);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. This
 * one-shot form loads and releases the encoder around a single image. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
void h3_video_latent_free(h3_video_latent *latent);

#endif
