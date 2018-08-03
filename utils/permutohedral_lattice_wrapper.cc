#include <vector>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "cuda_utils.h"
#include "permutohedral_lattice.h"
#include <THC.h>

void free_cuda_data_callback(void *p, void *data) {
  CUDA_CHECK_ERROR(cudaFree(data));
};

// inputs:
// position_in_tensor     (b, n_points_in, d)
// position_out_tensor    (b, n_points_out, d)
// features_in_tensor     (b, df_in, n_points_in)
// weights_tensor         (df_out, df_in, n_neighbors)
// neighbor_size          neighbor_size
// skip_conv  true:       splat-conv-slice
//            false:      splat-slice
//
// outputs:
// features_out_tensor    (b, df_out, n_points_out)
// lattices_in_tensor     (b, df_in, n_filled)
// offset_in_tensor       (b, n_points_in, d + 1)
// offset_out_tensor      (b, n_points_out, d + 1)
// barycentric_in_tensor  (b, n_points_in, d + 2)
// barycentric_out_tensor (b, n_points_out, d + 2)
// conv_neighbors_tensor  (b, n_neighbors, n_filled)
// norm_tensor            (b, 1, n_points_out)
std::vector<at::Tensor> permutohedral_lattice(at::Tensor position_in_tensor,
                                              at::Tensor position_out_tensor,
                                              at::Tensor features_in_tensor,
                                              at::Tensor weights_tensor,
                                              int neighbor_size,
                                              bool skip_conv) {
  CHECK_INPUT(position_in_tensor);
  CHECK_INPUT_TYPE(position_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(position_out_tensor);
  CHECK_INPUT_TYPE(position_out_tensor, at::ScalarType::Float);
  CHECK_INPUT(features_in_tensor);
  CHECK_INPUT_TYPE(features_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);

  int b = position_in_tensor.size(0);
  int n_points_in = position_in_tensor.size(1);
  int n_points_out = position_out_tensor.size(1);
  int d = position_in_tensor.size(2);
  int df_in = weights_tensor.size(1);
  int df_out = weights_tensor.size(0);
  int n_neighbors = static_cast<int>(pow(neighbor_size + 1, d + 1)) - static_cast<int>(pow(neighbor_size, d + 1));

  at::Tensor norm_features_tensor = torch::CUDA(at::kFloat).ones({b, 1, n_points_in});
  at::Tensor features_out_tensor = torch::CUDA(at::kFloat).zeros({b, df_out, n_points_out});
  at::Tensor offset_in_tensor = torch::CUDA(at::kInt).zeros({b, n_points_in, d + 1});
  at::Tensor offset_out_tensor = torch::CUDA(at::kInt).zeros({b, n_points_out, d + 1});
  at::Tensor barycentric_in_tensor = torch::CUDA(at::kFloat).zeros({b, n_points_in, d + 2});
  at::Tensor barycentric_out_tensor = torch::CUDA(at::kFloat).zeros({b, n_points_out, d + 2});
  at::Tensor norm_tensor = torch::CUDA(at::kFloat).zeros({b, 1, n_points_out});

  const float *position_in = position_in_tensor.data<float>();
  const float *position_out = position_out_tensor.data<float>();
  const float *features_in = features_in_tensor.data<float>();
  const float *norm_features = norm_features_tensor.data<float>();
  const float *weights = weights_tensor.data<float>();
  float *features_out = features_out_tensor.data<float>();
  int *offset_in = offset_in_tensor.data<int>();
  int *offset_out = offset_out_tensor.data<int>();
  float *barycentric_in = barycentric_in_tensor.data<float>();
  float *barycentric_out = barycentric_out_tensor.data<float>();
  float *norm = norm_tensor.data<float>();

  float *lattices_in;
  int *conv_neighbors;

  int n_filled = compute(b, n_points_in, n_points_out, df_in, df_out, d, neighbor_size, skip_conv, false,
                         position_in, position_out, features_in, norm_features, weights, features_out,
                         lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm);

  at::Tensor lattices_in_tensor =
      torch::CUDA(at::kFloat).tensorFromBlob(lattices_in, {b, df_in, n_filled},
                                             std::bind(free_cuda_data_callback, std::placeholders::_1, lattices_in));
  at::Tensor conv_neighbors_tensor =
      torch::CUDA(at::kInt).tensorFromBlob(conv_neighbors, {b, n_neighbors, n_filled},
                                           std::bind(free_cuda_data_callback, std::placeholders::_1, conv_neighbors));

  return {features_out_tensor,
          lattices_in_tensor,
          offset_in_tensor,
          offset_out_tensor,
          barycentric_in_tensor,
          barycentric_out_tensor,
          conv_neighbors_tensor,
          norm_tensor};
}

// inputs:
// grad_out_tensor        (b, df_out, n_points_out)
// weights_tensor         (df_out, df_in, n_neighbors)
// lattices_in_tensor     (b, df_in, n_filled)
// offset_in_tensor       (b, n_points_in, d + 1)
// offset_out_tensor      (b, n_points_out, d + 1)
// barycentric_in_tensor  (b, n_points_in, d + 2)
// barycentric_out_tensor (b, n_points_out, d + 2)
// conv_neighbors_tensor  (b, n_neighbors, n_filled)
// norm_tensor            (b, 1, n_points_out)
// neighbors_size         neighbors_size
// outputs:
// grad_in_tensor        (b, df_in, n_points_in)
// grad_weights_tensor   (df_out, df_in, n_neighbors)
std::vector<at::Tensor> permutohedral_lattice_grad(at::Tensor grad_out_tensor,
                                                   at::Tensor weights_tensor,
                                                   at::Tensor lattices_in_tensor,
                                                   at::Tensor offset_in_tensor,
                                                   at::Tensor offset_out_tensor,
                                                   at::Tensor barycentric_in_tensor,
                                                   at::Tensor barycentric_out_tensor,
                                                   at::Tensor conv_neighbors_tensor,
                                                   at::Tensor norm_tensor,
                                                   int neighbor_size,
                                                   bool skip_conv) {
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT_TYPE(grad_out_tensor, at::ScalarType::Float);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  CHECK_INPUT(lattices_in_tensor);
  CHECK_INPUT_TYPE(lattices_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(offset_in_tensor);
  CHECK_INPUT_TYPE(offset_in_tensor, at::ScalarType::Int);
  CHECK_INPUT(offset_out_tensor);
  CHECK_INPUT_TYPE(offset_out_tensor, at::ScalarType::Int);
  CHECK_INPUT(barycentric_in_tensor);
  CHECK_INPUT_TYPE(barycentric_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(barycentric_out_tensor);
  CHECK_INPUT_TYPE(barycentric_out_tensor, at::ScalarType::Float);
  CHECK_INPUT(conv_neighbors_tensor);
  CHECK_INPUT_TYPE(conv_neighbors_tensor, at::ScalarType::Int);
  CHECK_INPUT(norm_tensor);
  CHECK_INPUT_TYPE(norm_tensor, at::ScalarType::Float);

  int b = grad_out_tensor.size(0);
  int n_points_in = offset_in_tensor.size(1);
  int n_points_out = grad_out_tensor.size(2);
  int d = offset_in_tensor.size(2) - 1;
  int df_in = lattices_in_tensor.size(1);
  int df_out = grad_out_tensor.size(1);
  int n_filled = lattices_in_tensor.size(2);
  int n_neighbors = static_cast<int>(pow(neighbor_size + 1, d + 1)) - static_cast<int>(pow(neighbor_size, d + 1));

  at::Tensor weights_transpose_tensor = weights_tensor.transpose(0, 1).contiguous();
  at::Tensor grad_in_tensor = torch::CUDA(at::kFloat).zeros({b, df_in, n_points_in});
  at::Tensor grad_weights_transpose_tensor = torch::CUDA(at::kFloat).zeros({df_out, n_neighbors, df_in});

  const float *grad_out = grad_out_tensor.data<float>();
  const float *weights_transpose = weights_transpose_tensor.data<float>();
  const float *lattices_in = lattices_in_tensor.data<float>();
  const int *offset_in = offset_in_tensor.data<int>();
  const int *offset_out = offset_out_tensor.data<int>();
  const float *barycentric_in = barycentric_in_tensor.data<float>();
  const float *barycentric_out = barycentric_out_tensor.data<float>();
  const int *conv_neighbors = conv_neighbors_tensor.data<int>();
  const float *norm = norm_tensor.data<float>();
  float *grad_in = grad_in_tensor.data<float>();
  float *grad_weights_transpose = grad_weights_transpose_tensor.data<float>();

  compute_grad(b, n_points_in, n_points_out, n_filled, df_in, df_out, d, neighbor_size, skip_conv,
               grad_out, weights_transpose,
               lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm,
               grad_in, grad_weights_transpose);

  return {grad_in_tensor, grad_weights_transpose_tensor.transpose(1, 2).contiguous()};
}

// inputs:
// position_tensor        (b, n_points_in, d)
// features_in_tensor     (b, df_in, n_points_in)
// weights_tensor         (df_out, df_in, n_neighbors)
// neighbor_size          neighbor_size
// skip_conv  true:       splat-conv-slice
//            false:      splat-slice
//
// outputs:
// features_out_tensor    (b, df_out, n_points_out)
// lattices_in_tensor     (b, df_in, n_filled)
// offset_in_tensor       (b, n_points_in, d + 1)
// barycentric_in_tensor  (b, n_points_in, d + 2)
// conv_neighbors_tensor  (b, n_filled, n_neighbors)
// norm_tensor            (b, 1, n_points_out)
std::vector<at::Tensor> permutohedral_lattice_keep_position(at::Tensor position_in_tensor,
                                                            at::Tensor features_in_tensor,
                                                            at::Tensor weights_tensor,
                                                            int neighbor_size,
                                                            bool skip_conv) {
  CHECK_INPUT(position_in_tensor);
  CHECK_INPUT_TYPE(position_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(features_in_tensor);
  CHECK_INPUT_TYPE(features_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);

  int b = position_in_tensor.size(0);
  int n_points_in = position_in_tensor.size(1);
  int n_points_out = n_points_in;
  int d = position_in_tensor.size(2);
  int df_in = weights_tensor.size(1);
  int df_out = weights_tensor.size(0);
  int n_neighbors = static_cast<int>(pow(neighbor_size + 1, d + 1)) - static_cast<int>(pow(neighbor_size, d + 1));

  at::Tensor norm_features_tensor = torch::CUDA(at::kFloat).ones({b, 1, n_points_in});
  at::Tensor features_out_tensor = torch::CUDA(at::kFloat).zeros({b, df_out, n_points_out});
  at::Tensor offset_in_tensor = torch::CUDA(at::kInt).zeros({b, n_points_in, d + 1});
  at::Tensor barycentric_in_tensor = torch::CUDA(at::kFloat).zeros({b, n_points_in, d + 2});
  at::Tensor norm_tensor = torch::CUDA(at::kFloat).zeros({b, 1, n_points_out});

  const float *position_in = position_in_tensor.data<float>();
  const float *position_out = 0;
  const float *features_in = features_in_tensor.data<float>();
  const float *norm_features = norm_features_tensor.data<float>();
  const float *weights = weights_tensor.data<float>();
  float *features_out = features_out_tensor.data<float>();
  int *offset_in = offset_in_tensor.data<int>();
  int *offset_out = 0;
  float *barycentric_in = barycentric_in_tensor.data<float>();
  float *barycentric_out = 0;
  float *norm = norm_tensor.data<float>();

  float *lattices_in;
  int *conv_neighbors;

  int n_filled = compute(b, n_points_in, n_points_out, df_in, df_out, d, neighbor_size, skip_conv, true,
                         position_in, position_out, features_in, norm_features, weights, features_out,
                         lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm);

  at::Tensor lattices_in_tensor =
      torch::CUDA(at::kFloat).tensorFromBlob(lattices_in, {b, df_in, n_filled},
                                             std::bind(free_cuda_data_callback, std::placeholders::_1, lattices_in));
  at::Tensor conv_neighbors_tensor =
      torch::CUDA(at::kInt).tensorFromBlob(conv_neighbors, {b, n_neighbors, n_filled},
                                           std::bind(free_cuda_data_callback, std::placeholders::_1, conv_neighbors));

  return {features_out_tensor,
          lattices_in_tensor,
          offset_in_tensor,
          barycentric_in_tensor,
          conv_neighbors_tensor,
          norm_tensor};
}

// inputs:
// grad_out_tensor        (b, df_out, n_points_out)
// weights_tensor         (df_out, df_in, n_neighbors)
// lattices_in_tensor     (b, df_in, n_filled)
// offset_in_tensor       (b, n_points_in, d + 1)
// barycentric_in_tensor  (b, n_points_in, d + 2)
// conv_neighbors_tensor  (b, n_filled, n_neighbors)
// norm_tensor            (b, 1, n_points_out)
// neighbors_size         neighbors_size
// outputs:
// grad_in_tensor        (b, df_in, n_points_in)
// grad_weights_tensor   (df_out, df_in, n_neighbors)
std::vector<at::Tensor> permutohedral_lattice_keep_position_grad(at::Tensor grad_out_tensor,
                                                                 at::Tensor weights_tensor,
                                                                 at::Tensor lattices_in_tensor,
                                                                 at::Tensor offset_in_tensor,
                                                                 at::Tensor barycentric_in_tensor,
                                                                 at::Tensor conv_neighbors_tensor,
                                                                 at::Tensor norm_tensor,
                                                                 int neighbor_size,
                                                                 bool skip_conv) {
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT_TYPE(grad_out_tensor, at::ScalarType::Float);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  CHECK_INPUT(lattices_in_tensor);
  CHECK_INPUT_TYPE(lattices_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(offset_in_tensor);
  CHECK_INPUT_TYPE(offset_in_tensor, at::ScalarType::Int);
  CHECK_INPUT(barycentric_in_tensor);
  CHECK_INPUT_TYPE(barycentric_in_tensor, at::ScalarType::Float);
  CHECK_INPUT(conv_neighbors_tensor);
  CHECK_INPUT_TYPE(conv_neighbors_tensor, at::ScalarType::Int);
  CHECK_INPUT(norm_tensor);
  CHECK_INPUT_TYPE(norm_tensor, at::ScalarType::Float);

  int b = grad_out_tensor.size(0);
  int n_points_in = offset_in_tensor.size(1);
  int n_points_out = n_points_in;
  int d = offset_in_tensor.size(2) - 1;
  int df_in = lattices_in_tensor.size(1);
  int df_out = grad_out_tensor.size(1);
  int n_filled = lattices_in_tensor.size(2);
  int n_neighbors = static_cast<int>(pow(neighbor_size + 1, d + 1)) - static_cast<int>(pow(neighbor_size, d + 1));

  at::Tensor weights_transpose_tensor = weights_tensor.transpose(0, 1).contiguous();
  at::Tensor grad_in_tensor = torch::CUDA(at::kFloat).zeros({b, df_in, n_points_in});
  at::Tensor grad_weights_transpose_tensor = torch::CUDA(at::kFloat).zeros({df_out, n_neighbors, df_in});

  const float *grad_out = grad_out_tensor.data<float>();
  const float *weights_transpose = weights_transpose_tensor.data<float>();
  const float *lattices_in = lattices_in_tensor.data<float>();
  const int *offset_in = offset_in_tensor.data<int>();
  const int *offset_out = offset_in;
  const float *barycentric_in = barycentric_in_tensor.data<float>();
  const float *barycentric_out = barycentric_in;
  const int *conv_neighbors = conv_neighbors_tensor.data<int>();
  const float *norm = norm_tensor.data<float>();
  float *grad_in = grad_in_tensor.data<float>();
  float *grad_weights_transpose = grad_weights_transpose_tensor.data<float>();

  compute_grad(b, n_points_in, n_points_out, n_filled, df_in, df_out, d, neighbor_size, skip_conv,
               grad_out, weights_transpose,
               lattices_in, offset_in, offset_out, barycentric_in, barycentric_out, conv_neighbors, norm,
               grad_in, grad_weights_transpose);

  return {grad_in_tensor, grad_weights_transpose_tensor.transpose(1, 2).contiguous()};
}

PYBIND11_MODULE(C_, m) {
  m.def("permutohedral_lattice", &permutohedral_lattice, "permutohedral lattice");
  m.def("permutohedral_lattice_keep_position", &permutohedral_lattice_keep_position, "permutohedral lattice");
  m.def("permutohedral_lattice_grad", &permutohedral_lattice_grad, "permutohedral lattice grad");
  m.def("permutohedral_lattice_keep_position_grad", &permutohedral_lattice_keep_position_grad, "permutohedral lattice grad");
}
