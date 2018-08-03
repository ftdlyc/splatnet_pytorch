#ifndef PERMUTOHEDRAL_LATTICE_H
#define PERMUTOHEDRAL_LATTICE_H

int compute(
    int b, int n_points_in, int n_points_out, int df_in, int df_out, int d, int n, bool skip_conv, bool keep_position,
    const float *position_in,
    const float *position_out,
    const float *features_in,
    const float *norm_features,
    const float *weights,
    float *features_out,
    float *&lattices_in,
    int *offset_in,
    int *offset_out,
    float *barycentric_in,
    float *barycentric_out,
    int *&conv_neightbors,
    float *norm);

void compute_grad(
    int b, int n_points_in, int n_points_out, int n_filled, int df_in, int df_out, int d, int n, bool skip_conv,
    const float *grad_out,
    const float *weights_transpose,
    const float *lattices_in,
    const int *offset_in,
    const int *offset_out,
    const float *barycentric_in,
    const float *barycentric_out,
    const int *conv_neightbors,
    const float *norm,
    float *grad_in,
    float *grad_weights_transpose);

#endif