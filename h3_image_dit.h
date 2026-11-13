#ifndef H3_IMAGE_DIT_H
#define H3_IMAGE_DIT_H

#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

#define H3_IMAGE_DIT_STEPS 6
#define H3_IMAGE_DIT_CHANNELS 24
#define H3_IMAGE_DIT_PATCH 96
#define H3_IMAGE_DIT_ROPE_SCALE 32.0

typedef struct h3_image_dit h3_image_dit;

/* One reference image of an edit, expressed as the latent grid it encodes to.
 * Both sides must be even so the (1,2,2) patch grid is whole. */
typedef struct {
    int latent_height;
    int latent_width;
} h3_image_reference_grid;

typedef void (*h3_image_dit_progress)(const char *phase, int completed,
                                      int total, void *opaque);

/* Load MiniMax-H3-Image 2D DiT. transformer_directory holds the BF16 shards;
 * pdd_heads_path is the fused PDD heads safetensors file (or a directory that
 * contains it). Text is refined once at load. SSD streaming keeps two BF16
 * block matrix slots so a 64 GiB Mac need not resident-map ~62 GiB.
 * active_blocks is a plumbing diagnostic: pass 50 for the released model. */
h3_image_dit *h3_image_dit_load(
    const char *transformer_directory, const char *pdd_heads_path,
    const char *shader_source_path, const h3_text_embedding *text,
    int latent_height, int latent_width, int ssd_streaming, int active_blocks,
    const h3_image_reference_grid *references, size_t reference_count,
    h3_image_dit_progress progress, void *progress_opaque, char *error,
    size_t error_size);

size_t h3_image_dit_latent_elements(const h3_image_dit *dit);

/* normalized_latent is channel-major F32 [24, H, W]. On entry it is the
 * initial noise; on success it is overwritten with the denoised normalized
 * latent after the fixed six-step PDD schedule.
 *
 * For an edit, reference_latents holds one channel-major [24, h, w] tensor per
 * reference, already normalized and already noised to t = 0.999. Their rows
 * ride through every step unchanged and only the target rows are written. */
int h3_image_dit_denoise(h3_image_dit *dit, float *normalized_latent,
                         const float *const *reference_latents,
                         h3_image_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

void h3_image_dit_free(h3_image_dit *dit);

/* ---- Host helpers (cheap unit tests; no weights required) ---- */

/* Fill 7 sigmas (6 steps + terminal 0) after the float32 t=1-sigma round trip
 * used by the released sampler. */
int h3_image_dit_fixed_sigmas(float *sigmas, size_t count);

/* 2D patchify/unpatchify with patch (1,2,2): latent [C,H,W] <-> rows [N, C*4]. */
int h3_image_dit_patchify(const float *latent, int channels, int height,
                          int width, float *rows, size_t row_elements);
int h3_image_dit_unpatchify(const float *rows, int channels, int height,
                            int width, float *latent, size_t latent_elements);

/* Aspect-normalized spatial RoPE axis: dim//patch points on [left, left+ratio)
 * scaled by 32 (matches numpy linspace(..., endpoint=False)). */
int h3_image_dit_spatial_axis(int dim, int patch, double *axis,
                              size_t axis_count);

/* Build packed (t,h,w) positions for [text | image]: text t=0..N-1, image t=N,
 * spatial axes aspect-normalized * 32. positions has sequence*3 doubles in
 * row-major (t,h,w). */
int h3_image_dit_build_positions(size_t text_tokens, int latent_height,
                                 int latent_width, double *positions,
                                 size_t position_elements);

/* Edit variant: `[text | references | target]` rotary grid. Each reference
 * takes its own integer frame slot on the rotary clock. */
int h3_image_dit_build_positions_refs(size_t text_tokens,
                                      const h3_image_reference_grid *references,
                                      size_t reference_count,
                                      int latent_height, int latent_width,
                                      double *positions,
                                      size_t position_elements);

/* Analytic rope.inv_freq for rope_freq_dim=16, theta=10000. */
int h3_image_dit_rope_inv_freq(float *inv_freq, size_t count);

/* Euler velocity update used by Image PDD: x += (sigma - sigma_next) * v. */
void h3_image_dit_velocity_update(float *sample, const float *velocity,
                                  size_t elements, float sigma,
                                  float sigma_next);

#endif
