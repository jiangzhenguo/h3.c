#ifndef H3_IMAGE_VAE_H
#define H3_IMAGE_VAE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int width;
    int height;
    /* Row-major interleaved RGB8 after ImageNet denorm + truncate. */
    uint8_t *rgb;
    h3_gpu_stats gpu_stats;
} h3_image_vae_output;

typedef void (*h3_image_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);

/* Decode a channel-major F32 normalized latent [24, H, W]. Applies
 * latents_std/mean, post_quant_conv, the 24-layer 2D ViT decoder, ImageNet
 * denorm, and returns truncated RGB8. Spatial tiling (256/64) is on by
 * default so a latent that fits one tile matches the untiled path. */
int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_height,
                        int latent_width, h3_image_vae_progress progress,
                        void *progress_opaque, h3_image_vae_output *output,
                        char *error, size_t error_size);
void h3_image_vae_output_free(h3_image_vae_output *output);

/* ---- Host helpers (cheap unit tests; no weights required) ---- */

enum {
    H3_IMAGE_VAE_CHANNELS = 24,
    H3_IMAGE_VAE_PATCH = 16,
    H3_IMAGE_VAE_TILE_PIXELS = 256,
    H3_IMAGE_VAE_TILE_OVERLAP_MIN = 64,
    H3_IMAGE_VAE_ROPE_DIM = 32,
    H3_IMAGE_VAE_ROPE_FREQS = 8
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} h3_image_vae_tile_axis;

/* Pixel-space tile plan matching AutoencoderKLMiniMaxH3Image._split_tiles. */
int h3_image_vae_tile_axis_build(int extent, int tile_pixels, int min_overlap,
                                 int spatial_ratio,
                                 h3_image_vae_tile_axis *axis);
void h3_image_vae_tile_axis_free(h3_image_vae_tile_axis *axis);

/* Exact linear blend of overlapping RGB F32 tiles into a full canvas.
 * tiles is row-major over the y/x tile grid; each tile is
 * tile_h * tile_w * 3 floats. */
int h3_image_vae_blend_tiles(float **tiles,
                             const h3_image_vae_tile_axis *y_axis,
                             const h3_image_vae_tile_axis *x_axis,
                             float *rgb_out, size_t rgb_elements);

/* Length-normalized 2D RoPE axis: 2*((i+0.5)/size)-1 for i in [0, size). */
int h3_image_vae_rope_axis(int size, float *axis, size_t axis_count);

/* Analytic inv_freq for rope_dim=32, num_axes=2, theta=100. */
int h3_image_vae_rope_inv_freq(float *inv_freq, size_t count);

/* Pack (h,w) positions for patches + 5 suffix zeros into positions[seq*2]. */
int h3_image_vae_build_positions(int latent_height, int latent_width,
                                 float *positions, size_t position_elements);

/* Unpatch projected rows [H*W, 3*16*16] into channel-major [3, H*16, W*16]. */
int h3_image_vae_unpatch(const float *rows, int latent_height,
                         int latent_width, float *chw, size_t chw_elements);

/* latent[c,h,w] *= std[c]; latent[c,h,w] += mean[c]. */
void h3_image_vae_denormalize_latent(float *latent, int height, int width,
                                     const float *mean, const float *std);

/* ImageNet mean/std denorm of channel-major F32 RGB in [?,?], clamp [0,1],
 * truncate to RGB8 row-major. */
int h3_image_vae_imagenet_to_rgb8(const float *chw, int height, int width,
                                  uint8_t *rgb, size_t rgb_elements);

#endif
