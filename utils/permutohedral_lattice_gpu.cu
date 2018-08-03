#include <utility>
#include <vector>
#include "cuda_utils.h"
#include "cuda_headers.h"
#include "permutohedral_lattice.h"

//----------------------------------------------------------------------------------------//
//-------------------------------------- Hash Table --------------------------------------//
//----------------------------------------------------------------------------------------//

typedef struct HashTable {
  size_t *key_size;
  size_t *filled;
  size_t *capacity;
  int16_t *keys;
  int *table;
} HashTable;

__host__ void init_hash_table(int b, size_t key_size, size_t capacity,
                              HashTable **hash_tables_host, HashTable **hash_tables_gpu) {
  size_t filled = 0;
  *hash_tables_host = new HashTable[b];
  for (int i = 0; i < b; ++i) {
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&(*hash_tables_host)[i].key_size), sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&(*hash_tables_host)[i].filled), sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&(*hash_tables_host)[i].capacity), sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&(*hash_tables_host)[i].keys), capacity * key_size * sizeof(int16_t)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&(*hash_tables_host)[i].table), capacity * sizeof(int)));

    CUDA_CHECK_ERROR(cudaMemcpy((*hash_tables_host)[i].key_size, &key_size, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy((*hash_tables_host)[i].filled, &filled, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy((*hash_tables_host)[i].capacity, &capacity, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset((*hash_tables_host)[i].keys, 0, capacity * key_size * sizeof(int16_t)));
    CUDA_CHECK_ERROR(cudaMemset((*hash_tables_host)[i].table, -1, capacity * sizeof(int)));
  }
  CUDA_CHECK_ERROR(cudaMalloc((void **) hash_tables_gpu, b * sizeof(HashTable)));
  CUDA_CHECK_ERROR(cudaMemcpy(*hash_tables_gpu, *hash_tables_host, b * sizeof(HashTable), cudaMemcpyHostToDevice));
};

__host__ void deinit_hash_table(int b, HashTable *hash_tables_host, HashTable *hash_tables_gpu) {
  for (int i = 0; i < b; ++i) {
    CUDA_CHECK_ERROR(cudaFree(hash_tables_host[i].key_size));
    CUDA_CHECK_ERROR(cudaFree(hash_tables_host[i].filled));
    CUDA_CHECK_ERROR(cudaFree(hash_tables_host[i].capacity));
    CUDA_CHECK_ERROR(cudaFree(hash_tables_host[i].keys));
    CUDA_CHECK_ERROR(cudaFree(hash_tables_host[i].table));
  }
  CUDA_CHECK_ERROR(cudaFree(hash_tables_gpu));
  delete[] hash_tables_host;
};

__device__ __host__ size_t hash(HashTable &hash_table, const int16_t *key) {
  size_t s = 0;
  for (int i = 0; i < *hash_table.key_size; ++i) {
    s += key[i];
    s *= 2531011;
  }
  return s;
}

__device__ __host__ int find_hash(HashTable &hash_table, const int16_t *key, bool create = false) {
  size_t &key_size = *hash_table.key_size;
  size_t &filled = *hash_table.filled;
  size_t &capacity = *hash_table.capacity;

  size_t h = hash(hash_table, key) % capacity;
  while (true) {
    int e = hash_table.table[h];
    if (e == -1) {
      if (create && filled < capacity) {
        for (int i = 0; i < key_size; i++) {
          hash_table.keys[filled * key_size + i] = key[i];
        }
        hash_table.table[h] = static_cast<int>(filled++);
        return hash_table.table[h];
      }
      return -1;
    }

    bool good = true;
    for (int i = 0; i < key_size; ++i) {
      good &= (hash_table.keys[e * key_size + i] == key[i]);
    }
    if (good) {
      return e;
    }

    h = h < capacity - 1 ? h + 1 : 0;
  }
}


//----------------------------------------------------------------------------------------//
//----------------------------- Compute neighbors And Gauss -----------------------------//
//----------------------------------------------------------------------------------------//

class PermutationGridCorCallBack {
 public:
  PermutationGridCorCallBack(int d1, int n_grid_cor, int16_t *grid_)
      : n_(0), d1_(d1), n_grid_cor_(n_grid_cor), grid_(grid_) {}

  void operator()(int16_t *grid_cor) {
    memcpy(grid_, grid_cor, d1_ * sizeof(int16_t));
    ++n_;
    if (n_ < n_grid_cor_) {
      grid_ += d1_;
    }
  }

  int n_;
  int d1_;
  int n_grid_cor_;
  int16_t *grid_;
};

__host__ void walk_in_dimension(int d, int dimension, int step, int16_t *key) {
  for (int i = 0; i < d + 1; ++i) {
    key[i] -= step;
  }
  key[dimension] += step * d;
}

__host__ void permutation_grid_cor(int start, int end, int n, int16_t *grid_cor, PermutationGridCorCallBack &yield) {
  if (start == end) {
    yield(grid_cor);
    return;
  }
  for (int i = 0; i < n + 1; ++i) {
    grid_cor[start] = i;
    permutation_grid_cor(start + 1, end, n, grid_cor, yield);
  }
}

__host__ void compute_neighbors_and_gauss(int d, int n, int16_t *neighbors, float *gauss_weights) {
  // compute convolution filter relative position
  int d1 = d + 1;
  int n_grid_cor = static_cast<int>(pow(n + 1, d1));
  int16_t *grid = new int16_t[n_grid_cor * d1];
  int16_t *grid_cor = new int16_t[d1];
  PermutationGridCorCallBack yield(d1, n_grid_cor, grid);
  permutation_grid_cor(0, d1, n, grid_cor, yield);

  HashTable hash_table{new size_t[1], new size_t[1], new size_t[1], new int16_t[n_grid_cor * d1], new int[n_grid_cor]};
  *hash_table.key_size = d1;
  *hash_table.filled = 0;
  *hash_table.capacity = n_grid_cor;
  memset(hash_table.keys, 0, n_grid_cor * d1 * sizeof(int16_t));
  memset(hash_table.table, -1, n_grid_cor * sizeof(int));

  int16_t *lattice_cor = new int16_t[d1];
  int16_t *b = new int16_t[d1 * d1];

  memset(b, -1, d1 * d1 * sizeof(int16_t));
  for (int i = 0; i < d1; ++i) {
    b[i * d1 + i] = d1 - 1;
  }
  for (int i = 0; i < n_grid_cor; ++i) {
    memset(lattice_cor, 0, d1 * sizeof(int16_t));
    for (int j = 0; j < d1; ++j) {
      for (int k = 0; k < d1; ++k) {
        lattice_cor[j] += b[(j * d1 + k)] * grid[i * d1 + k];
      }
    }
    if (find_hash(hash_table, lattice_cor) == -1) {
      memcpy(neighbors + (*hash_table.filled) * d1, lattice_cor, d1 * sizeof(int16_t));
      find_hash(hash_table, lattice_cor, true);
    }
  }

  int n_neighbors = static_cast<int>(pow(n + 1, d1)) - static_cast<int>(pow(n, d1));
  assert((*hash_table.filled) == n_neighbors);

  // compute gauss filter weights
  std::vector<float> filter{1.0, 0.5};
  float *gauss_weights_tmp = new float[n_neighbors];
  int16_t *walking_key_up = new int16_t[d1];
  int16_t *walking_key_down = new int16_t[d1];
  memset(gauss_weights, 0, n_neighbors * sizeof(float));
  gauss_weights[0] = 1;
  for (int i = 0; i < d1; ++i) {
    memset(gauss_weights_tmp, 0, n_neighbors * sizeof(float));

    for (int j = 0; j < n_neighbors; ++j) {
      const int16_t *key = hash_table.keys + j * d1;
      memcpy(walking_key_up, key, d1 * sizeof(int16_t));
      memcpy(walking_key_down, key, d1 * sizeof(int16_t));

      float &v = gauss_weights_tmp[j];
      v += gauss_weights[j] * filter[0];
      for (int k = 1; k <= n && k <= 2; ++k) {
        walk_in_dimension(d1, i, 1, walking_key_up);
        walk_in_dimension(d1, i, -1, walking_key_down);

        int h1 = find_hash(hash_table, walking_key_up);
        int h2 = find_hash(hash_table, walking_key_down);

        v += ((h1 >= 0 ? gauss_weights[h1] : 0) +
            (h2 >= 0 ? gauss_weights[h2] : 0)) * (k < filter.size() ? filter[k] : 0);
      }
    }
    memmove(gauss_weights, gauss_weights_tmp, n_neighbors * sizeof(float));
  }
  float norm_coef = gauss_weights[0];
  for (int i = 0; i < n_neighbors; ++i) {
    gauss_weights[i] /= norm_coef;
  }

  delete[] hash_table.key_size, hash_table.filled, hash_table.capacity, hash_table.keys, hash_table.table;
  delete[] grid, grid_cor, lattice_cor, b;
  delete[] gauss_weights_tmp, walking_key_up, walking_key_down;
}


//----------------------------------------------------------------------------------------//
//-------------------------------------- Initialize --------------------------------------//
//----------------------------------------------------------------------------------------//

// inputs: position_in (b, n_points, d)
//         keys        (b, n_points, d + 1, d)
//         barycentric (b, n_points, d + 2)
__global__ void init1_kernel(int d, int n_points,
                             const float *position,
                             int16_t *keys,
                             float *barycentric) {
  int batch_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int d1 = d + 1;

  for (int point_idx = thread_idx; point_idx < n_points; point_idx += blockDim.x) {
    const float *position_ = position + (batch_idx * n_points + point_idx) * d;
    int16_t *key_ = keys + (batch_idx * n_points + point_idx) * (d + 1) * d;
    float *barycentric_ = barycentric + (batch_idx * n_points + point_idx) * (d + 2);

    int16_t *canonical = new int16_t[d1 * d1];
    float *scale_factor = new float[d];
    float *elevated = new float[d1];
    int16_t *rem0 = new int16_t[d1];
    int16_t *rank = new int16_t[d1]{0};

    for (int i = 0; i < d1; ++i) {
      for (int j = 0; j < d1 - i; ++j) {
        canonical[i * d1 + j] = i;
      }
      for (int j = d1 - i; j < d1; ++j) {
        canonical[i * d1 + j] = -d1 + i;
      }
    }

    float inv_std_dev = sqrtf(2.0 / 3.0) * d1;
    for (int i = 0; i < d; ++i) {
      scale_factor[i] = 1.0 / sqrtf(((i + 1) * (i + 2))) * inv_std_dev;
    }

    // Elevate point into Hd using the rotation matrix E
    // see p.30 in [Adams etal 2011]
    float sm = 0;
    for (int i = d; i > 0; --i) {
      float cf = position_[i - 1] * scale_factor[i - 1];
      elevated[i] = sm - i * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    int sum = 0;
    for (int i = 0; i < d1; ++i) {
      int rd = static_cast<int>(round(elevated[i] / d1));
      rem0[i] = static_cast<int16_t>(rd * d1);
      sum += rd;
    }

    // Find the simplex we are in and store it in rank
    // (where rank describes what position coorinate i has in the sorted order of the features values)
    for (int i = 0; i < d; ++i) {
      float di = elevated[i] - rem0[i];
      for (int j = i + 1; j < d1; ++j) {
        if (di < elevated[j] - rem0[j]) {
          ++rank[i];
        } else {
          ++rank[j];
        }
      }
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i < d1; ++i) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d + 1;
        rem0[i] += d + 1;
      } else if (rank[i] > d) {
        rank[i] -= d + 1;
        rem0[i] -= d + 1;
      }
    }

    // Compute all vertices
    for (int remainder = 0; remainder < d1; ++remainder) {
      // all but the last coordinate - it's redundant because they sum to zero
      for (int i = 0; i < d; ++i) {
        key_[remainder * d + i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
      }
    }

    // comptue the barycentric coordinates
    // see p.31 in [Adams etal 2011]
    for (int i = 0; i < d + 2; ++i) {
      barycentric_[i] = 0;
    }
    for (int i = 0; i < d1; ++i) {
      float v = (elevated[i] - rem0[i]) / d1;
      barycentric_[d - rank[i]] += v;
      barycentric_[d - rank[i] + 1] -= v;
    }
    barycentric_[0] += 1.0 + barycentric_[d + 1];

    delete[] canonical;
    delete[] scale_factor;
    delete[] elevated;
    delete[] rem0;
    delete[] rank;
  }
}

// inputs: hash_table     (b, d)
//         keys           (b, n_points, d + 1, d)
//         offset         (b, n_points, d + 1)
__global__ void init2_kernel(int d, int n_points,
                             HashTable *hash_tables,
                             const int16_t *keys,
                             int *offset) {
  int batch_idx = blockIdx.x;
  HashTable &hash_table_ = hash_tables[batch_idx];
  int d1 = d + 1;

  // Compute all offset
  for (int i = 0; i < n_points; ++i) {
    for (int j = 0; j < d1; ++j) {
      const int16_t *key_ = keys + ((batch_idx * n_points + i) * (d + 1) + j) * d;
      int h = find_hash(hash_table_, key_, true);
      offset[(batch_idx * n_points + i) * (d + 1) + j] = h;
    }
  }
}

// inputs: hash_table     (b, d)
//         neighbors      (n_neighbors)
//         conv_neighbors (b, n_neighbors, n_filled)
__global__ void init3_kernel(int d, int n_neighbors, int n_filled,
                             HashTable *hash_tables,
                             const int16_t *neighbors,
                             int *conv_neighbors) {
  int batch_idx = blockIdx.x;
  HashTable &hash_table_ = hash_tables[batch_idx];
  int d1 = d + 1;

  // Compute all convlution neighbors
  size_t key_size = (*hash_table_.key_size);
  int n_filled_ = (*hash_table_.filled);
  for (int lattice_idx = 0; lattice_idx < n_filled_; ++lattice_idx) {
    int16_t *center = new int16_t[d1];
    int16_t sum = 0;
    for (int i = 0; i < d; ++i) {
      center[i] = hash_table_.keys[lattice_idx * key_size + i];
      sum += center[i];
    }
    center[d] = -sum;

    int16_t *neighbor_key = new int16_t[d1];
    for (int i = 0; i < n_neighbors; ++i) {
      for (int j = 0; j < d1; ++j) {
        neighbor_key[j] = center[j] + neighbors[i * d1 + j];
      }
      conv_neighbors[(batch_idx * n_neighbors + i) * n_filled + lattice_idx] = find_hash(hash_table_, neighbor_key);
    }

    delete[] center;
    delete[] neighbor_key;
  }
  for (int lattice_idx = n_filled_; lattice_idx < n_filled; ++lattice_idx) {
    for (int i = 0; i < n_neighbors; ++i) {
      conv_neighbors[(batch_idx * n_neighbors + i) * n_filled + lattice_idx] = -1;
    }
  }
}

// inputs: position_in (b, n_points, d)
//         hash_table  (b, d)
//         offset      (b, n_points, d + 1)
//         barycentric (b, n_points, d + 2)
__global__ void init4_kernel(int d, int n_points,
                             const float *position,
                             HashTable *hash_tables,
                             int *offset,
                             float *barycentric) {
  int batch_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  HashTable &hash_table_ = hash_tables[batch_idx];
  int d1 = d + 1;

  for (int point_idx = thread_idx; point_idx < n_points; point_idx += blockDim.x) {
    const float *position_ = position + (batch_idx * n_points + point_idx) * d;
    float *barycentric_ = barycentric + (batch_idx * n_points + point_idx) * (d + 2);

    int16_t *canonical = new int16_t[d1 * d1];
    float *scale_factor = new float[d];
    float *elevated = new float[d1];
    int16_t *rem0 = new int16_t[d1];
    int16_t *rank = new int16_t[d1]{0};
    int16_t *key = new int16_t[d];

    for (int i = 0; i < d1; ++i) {
      for (int j = 0; j < d1 - i; ++j) {
        canonical[i * d1 + j] = i;
      }
      for (int j = d1 - i; j < d1; ++j) {
        canonical[i * d1 + j] = -d1 + i;
      }
    }

    float inv_std_dev = sqrtf(2.0 / 3.0) * d1;
    for (int i = 0; i < d; ++i) {
      scale_factor[i] = 1.0 / sqrtf(((i + 1) * (i + 2))) * inv_std_dev;
    }

    // Elevate point into Hd using the rotation matrix E
    // see p.30 in [Adams etal 2011]
    float sm = 0;
    for (int i = d; i > 0; --i) {
      float cf = position_[i - 1] * scale_factor[i - 1];
      elevated[i] = sm - i * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    int sum = 0;
    for (int i = 0; i < d1; ++i) {
      int rd = static_cast<int>(round(elevated[i] / d1));
      rem0[i] = static_cast<int16_t>(rd * d1);
      sum += rd;
    }

    // Find the simplex we are in and store it in rank
    // (where rank describes what position coorinate i has in the sorted order of the features values)
    for (int i = 0; i < d; ++i) {
      float di = elevated[i] - rem0[i];
      for (int j = i + 1; j < d1; ++j) {
        if (di < elevated[j] - rem0[j]) {
          ++rank[i];
        } else {
          ++rank[j];
        }
      }
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i < d1; ++i) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d + 1;
        rem0[i] += d + 1;
      } else if (rank[i] > d) {
        rank[i] -= d + 1;
        rem0[i] -= d + 1;
      }
    }

    // Compute all offset
    for (int remainder = 0; remainder < d1; ++remainder) {
      // all but the last coordinate - it's redundant because they sum to zero
      for (int i = 0; i < d; ++i) {
        key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
      }
      int h = find_hash(hash_table_, key);
      offset[(batch_idx * n_points + point_idx) * (d + 1) + remainder] = h;
    }

    // comptue the barycentric coordinates
    // see p.31 in [Adams etal 2011]
    for (int i = 0; i < d + 2; ++i) {
      barycentric_[i] = 0;
    }
    for (int i = 0; i < d1; ++i) {
      float v = (elevated[i] - rem0[i]) / d1;
      barycentric_[d - rank[i]] += v;
      barycentric_[d - rank[i] + 1] -= v;
    }
    barycentric_[0] += 1.0 + barycentric_[d + 1];

    delete[] canonical;
    delete[] scale_factor;
    delete[] elevated;
    delete[] rem0;
    delete[] rank;
    delete[] key;
  }
}

//----------------------------------------------------------------------------------------//
//-------------------------------------- Operation ---------------------------------------//
//----------------------------------------------------------------------------------------//

// inputs: features    (b, df, n_points)
//         offset      (b, n_points, d + 1)
//         barycentric (b, n_points, d + 2)
//         lattices    (b, df, n_filled)
__global__ void splat_kernel(int d, int df, int n_points, int n_filled,
                             const float *features,
                             const int *offset,
                             const float *barycentric,
                             float *lattices) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_points; j += blockDim.x) {
      float feature_ = features[(batch_idx * df + i) * n_points + j];

      for (int k = 0; k < (d + 1); ++k) {
        int offset_ = offset[(batch_idx * n_points + j) * (d + 1) + k];
        float barycentric_ = barycentric[(batch_idx * n_points + j) * (d + 2) + k];
        atomicAdd(lattices + (batch_idx * df + i) * n_filled + offset_, feature_ * barycentric_);
      }
    }
  }
}

//inputs: lattices_in    (b, df_in, n_filled)
//        conv_weights   (df_out, df_in, n_neighbors)
//        conv_neighbors (b, n_neighbors, n_filled)
//        lattices_out   (b, df_out, n_filled)
__global__ void conv_kernel(int df_in, int df_out, int n_filled, int n_neighbors,
                            const float *lattices_in,
                            const float *conv_weights,
                            const int *conv_neighbors,
                            float *lattices_out) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df_out; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_filled; j += blockDim.x) {

      float sum = 0;
      for (int k = 0; k < df_in; ++k) {
        const float *conv_weights_ = conv_weights + (i * df_in + k) * n_neighbors;
        const float *lattices_in_ = lattices_in + (batch_idx * df_in + k) * n_filled;
        for (int l = 0; l < n_neighbors; ++l) {
          int h = conv_neighbors[(batch_idx * n_neighbors + l) * n_filled + j];
          if (h == -1) { continue; }
          sum += conv_weights_[l] * lattices_in_[h];
        }
      }
      lattices_out[(batch_idx * df_out + i) * n_filled + j] = sum;
    }
  }
}

// inputs: lattices       (b, df_in, n_filled)
//         conv_neighbors (b, n_neighbors, n_filled)
//         col            (df * n_neighbors, n_filled)
__global__ void img2col_kernel(int b, int df_in, int n_filled, int n_neighbors,
                               const float *lattices,
                               const int *conv_neighbors,
                               float *col) {
  int df_in_index = blockIdx.x;
  const float *lattices_ = lattices + (b * df_in + df_in_index) * n_filled;

  for (int i = threadIdx.x; i < n_filled; i += blockDim.x) {
    for (int j = 0; j < n_neighbors; ++j) {
      int h = conv_neighbors[(b * n_neighbors + j) * n_filled + i];
      if (h == -1) { continue; }
      col[(df_in_index * n_neighbors + j) * n_filled + i] = lattices_[h];
    }
  }
}

// inputs: lattices    (b, df, n_filled)
//         offset      (b, n_points, d + 1)
//         barycentric (b, n_points, d + 2)
//         features    (b, df, n_points)
__global__ void slice_kernel(int d, int df, int n_points, int n_filled,
                             const float *lattices,
                             const int *offset,
                             const float *barycentric,
                             float *features) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_points; j += blockDim.x) {

      for (int k = 0; k < (d + 1); ++k) {
        int offset_ = offset[(batch_idx * n_points + j) * (d + 1) + k];
        if (offset_ == -1) { continue; }
        float barycentric_ = barycentric[(batch_idx * n_points + j) * (d + 2) + k];
        float lattices_ = lattices[(batch_idx * df + i) * n_filled + offset_];
        atomicAdd(features + (batch_idx * df + i) * n_points + j, lattices_ * barycentric_);
      }
    }
  }
}

// inputs: grad_out    (b, df, n_filled)
//         offset      (b, n_points, d + 1)
//         barycentric (b, n_points, d + 2)
//         grad_in     (b, df, n_points)
__global__ void splat_grad_kernel(int d, int df, int n_points, int n_filled,
                                  const float *grad_out,
                                  const int *offset,
                                  const float *barycentric,
                                  float *grad_in) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_points; j += blockDim.x) {
      float &grad_in_ = grad_in[(batch_idx * df + i) * n_points + j];

      for (int k = 0; k < (d + 1); ++k) {
        int offset_ = offset[(batch_idx * n_points + j) * (d + 1) + k];
        float barycentric_ = barycentric[(batch_idx * n_points + j) * (d + 2) + k];
        float grad_out_ = grad_out[(batch_idx * df + i) * n_filled + offset_];
        grad_in_ += barycentric_ * grad_out_;
      }
    }
  }
}

//inputs: grad_out       (b, df_out, n_filled)
//        conv_weights   (df_out, df_in, n_neighbors)
//        conv_neighbors (b, n_neighbors, n_filled)
//        grad_in        (b, df_in, n_filled)
__global__ void conv_grad_kernel(int df_in, int df_out, int n_filled, int n_neighbors,
                                 const float *grad_out,
                                 const float *conv_weights,
                                 const int *conv_neighbors,
                                 float *grad_in) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df_out; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_filled; j += blockDim.x) {
      float grad_out_ = grad_out[(batch_idx * df_out + i) * n_filled + j];

      for (int k = 0; k < df_in; ++k) {
        const float *conv_weights_ = conv_weights + (i * df_in + k) * n_neighbors;
        float *grad_in_ = grad_in + (batch_idx * df_in + k) * n_filled;
        for (int l = 0; l < n_neighbors; ++l) {
          int h = conv_neighbors[(batch_idx * n_neighbors + l) * n_filled + j];
          if (h == -1) { continue; }
          atomicAdd(grad_in_ + h, conv_weights_[l] * grad_out_);
        }
      }
    }
  }
}

//inputs: grad_out       (b, df_out, n_filled)
//        conv_neighbors (b, n_neighbors, n_filled)
//        col_grad       (df_out * n_neighbors, n_filled)
__global__ void img2col_grad_kernel(int b, int df_out, int n_filled, int n_neighbors,
                                    const float *grad_out,
                                    const int *conv_neighbors,
                                    float *col_grad) {
  int df_out_index = blockIdx.x;
  const float *grad_out_ = grad_out + (b * df_out + df_out_index) * n_filled;

  for (int i = threadIdx.x; i < n_filled; i += blockDim.x) {
    for (int j = 0; j < n_neighbors; ++j) {
      int h = conv_neighbors[(b * n_neighbors + j) * n_filled + i];
      if (h == -1) { continue; }
      col_grad[(df_out_index * n_neighbors + j) * n_filled + i] = grad_out_[h];
    }
  }
}

// inputs: grad_out    (b, df, n_points)
//         offset      (b, n_points, d + 1)
//         barycentric (b, n_points, d + 2)
//         grad_in     (b, df, n_filled)
__global__ void slice_grad_kernel(int d, int df, int n_points, int n_filled,
                                  const float *grad_out,
                                  const int *offset,
                                  const float *barycentric,
                                  float *grad_in) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_points; j += blockDim.x) {

      for (int k = 0; k < (d + 1); ++k) {
        int offset_ = offset[(batch_idx * n_points + j) * (d + 1) + k];
        if (offset_ == -1) { continue; }
        float barycentric_ = barycentric[(batch_idx * n_points + j) * (d + 2) + k];
        float grad_out_ = grad_out[(batch_idx * df + i) * n_points + j];
        atomicAdd(grad_in + (batch_idx * df + i) * n_filled + offset_, grad_out_ * barycentric_);
      }
    }
  }
}

//inputs: grad_out       (b, df_out, n_filled)
//        lattices_in    (b, df_in, n_filled)
//        conv_neighbors (b, n_neighbors, n_filled)
//        grad_weights   (df_out, df_in, n_neighbors)
__global__ void weights_grad_kernel(int df_in, int df_out, int n_filled, int n_neighbors,
                                    const float *grad_out,
                                    const float *lattices_in,
                                    const int *conv_neighbors,
                                    float *grad_weights) {
  int batch_idx = blockIdx.x;

  for (int i = threadIdx.y; i < df_out; i += blockDim.y) {
    for (int j = threadIdx.x; j < n_filled; j += blockDim.x) {
      float grad_out_ = grad_out[(batch_idx * df_out + i) * n_filled + j];

      for (int k = 0; k < df_in; ++k) {
        const float *lattices_in_ = lattices_in + (batch_idx * df_in + k) * n_filled;
        for (int l = 0; l < n_neighbors; ++l) {
          int h = conv_neighbors[(batch_idx * n_neighbors + l) * n_filled + j];
          if (h == -1) { continue; }
          atomicAdd(grad_weights + (i * df_in + k) * n_neighbors + l, lattices_in_[h] * grad_out_);
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------//
//-------------------------------------- Operation ---------------------------------------//
//----------------------------------------------------------------------------------------//

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
    int *&conv_neighbors,
    float *norm) {
  int d1 = d + 1;
  int n_neighbors = static_cast<int>(pow(n + 1, d1)) - static_cast<int>(pow(n, d1));
  int n_filled = 0;
  size_t n_filled_tmp = 0;
  HashTable *hash_tables, *hash_tables_host;
  int16_t *keys_in, *neighbors, *neighbors_host;
  float *gauss_weights, *gauss_weights_host, *lattices_out;
  float *norm_lattices_in, *norm_lattices_out;

  // initialize
  neighbors_host = new int16_t[n_neighbors * d1];
  gauss_weights_host = new float[n_neighbors];
  compute_neighbors_and_gauss(d, n, neighbors_host, gauss_weights_host);

  init_hash_table(b, d, n_points_in * d1 * 10, &hash_tables_host, &hash_tables);
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&keys_in), b * n_points_in * d1 * d * sizeof(int16_t)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&neighbors), n_neighbors * d1 * sizeof(int16_t)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&gauss_weights), n_neighbors * sizeof(float)));
  CUDA_CHECK_ERROR(cudaMemcpy(neighbors, neighbors_host, n_neighbors * d1 * sizeof(int16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(gauss_weights, gauss_weights_host, n_neighbors * sizeof(float), cudaMemcpyHostToDevice));

  init1_kernel << < b, opt_n_threads(n_points_in) >> > (d, n_points_in, position_in, keys_in, barycentric_in);
  CUDA_CHECK_KERNEL_ERROR();
  init2_kernel << < b, 1 >> > (d, n_points_in, hash_tables, keys_in, offset_in);
  CUDA_CHECK_KERNEL_ERROR();
  for (int i = 0; i < b; ++i) {
    CUDA_CHECK_ERROR(cudaMemcpy(&n_filled_tmp, hash_tables_host[i].filled, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (n_filled_tmp > n_filled) {
      n_filled = n_filled_tmp;
    }
  }

  CUDA_CHECK_ERROR(cudaMallocManaged((void **) (&conv_neighbors), b * n_filled * n_neighbors * sizeof(int)));
  CUDA_CHECK_ERROR(cudaMallocManaged((void **) (&lattices_in), b * df_in * n_filled * sizeof(float)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&lattices_out), b * df_out * n_filled * sizeof(float)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&norm_lattices_in), b * 1 * n_filled * sizeof(float)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&norm_lattices_out), b * 1 * n_filled * sizeof(float)));

  init3_kernel << < b, 1 >> > (d, n_neighbors, n_filled, hash_tables, neighbors, conv_neighbors);
  CUDA_CHECK_KERNEL_ERROR();
  if (keep_position) {
    offset_out = offset_in;
    barycentric_out = barycentric_in;
  } else {
    init4_kernel << < b, opt_n_threads(n_points_out) >> >
        (d, n_points_out, position_out, hash_tables, offset_out, barycentric_out);
    CUDA_CHECK_KERNEL_ERROR();
  }

  // splat-conv-slice
  splat_kernel << < b, opt_block_config(n_points_in, df_in) >> >
      (d, df_in, n_points_in, n_filled, features_in, offset_in, barycentric_in, lattices_in);
  CUDA_CHECK_KERNEL_ERROR();
  if (skip_conv) {
    CUDA_CHECK_ERROR(
        cudaMemcpy(lattices_out, lattices_in, b * df_out * n_filled * sizeof(float), cudaMemcpyDeviceToDevice));
  } else {
//    // original convolution operation is too slow
//    conv_kernel << < b, opt_block_config(n_filled, df_out) >> >
//        (df_in, df_out, n_filled, n_neighbors, lattices_in, weights, conv_neighbors, lattices_out);
//    CUDA_CHECK_KERNEL_ERROR();

    // conv -> matmul
    float alpha = 1.0;
    float beta = 0;
    float *col;
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK_ERROR(cublasCreate(&cublas_handle));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&col), df_in * n_neighbors * n_filled * sizeof(float)));
    for (int i = 0; i < b; ++i) {
      CUDA_CHECK_ERROR(cudaMemset(col, 0, df_in * n_neighbors * n_filled * sizeof(float)));
      img2col_kernel << < df_in, opt_n_threads(n_filled) >> >
          (i, df_in, n_filled, n_neighbors, lattices_in, conv_neighbors, col);
      CUDA_CHECK_KERNEL_ERROR();
      CUBLAS_CHECK_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n_filled, df_out, df_in * n_neighbors,
                                     &alpha, col, n_filled, weights, df_in * n_neighbors,
                                     &beta, lattices_out + i * df_out * n_filled, n_filled));
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(cudaFree(col));
    CUBLAS_CHECK_ERROR(cublasDestroy(cublas_handle));
  }
  slice_kernel << < b, opt_block_config(n_points_out, df_out) >> >
      (d, df_out, n_points_out, n_filled, lattices_out, offset_out, barycentric_out, features_out);
  CUDA_CHECK_KERNEL_ERROR();

  // norm
  splat_kernel << < b, opt_block_config(n_points_in, 1) >> >
      (d, 1, n_points_in, n_filled, norm_features, offset_in, barycentric_in, norm_lattices_in);
  CUDA_CHECK_KERNEL_ERROR();
  conv_kernel << < b, opt_block_config(n_filled, 1) >> >
      (1, 1, n_filled, n_neighbors, norm_lattices_in, gauss_weights, conv_neighbors, norm_lattices_out);
  CUDA_CHECK_KERNEL_ERROR();
  slice_kernel << < b, opt_block_config(n_points_out, 1) >> >
      (d, 1, n_points_out, n_filled, norm_lattices_out, offset_out, barycentric_out, norm);
  CUDA_CHECK_KERNEL_ERROR();

  // deinitialize
  delete[] neighbors_host, gauss_weights_host;
  deinit_hash_table(b, hash_tables_host, hash_tables);
  CUDA_CHECK_ERROR(cudaFree(keys_in));
  CUDA_CHECK_ERROR(cudaFree(neighbors));
  CUDA_CHECK_ERROR(cudaFree(gauss_weights));
  CUDA_CHECK_ERROR(cudaFree(lattices_out));
  CUDA_CHECK_ERROR(cudaFree(norm_lattices_in));
  CUDA_CHECK_ERROR(cudaFree(norm_lattices_out));

  return n_filled;
}

void compute_grad(
    int b, int n_points_in, int n_points_out, int n_filled, int df_in, int df_out, int d, int n, bool skip_conv,
    const float *grad_out,
    const float *weights_transpose,
    const float *lattices_in,
    const int *offset_in,
    const int *offset_out,
    const float *barycentric_in,
    const float *barycentric_out,
    const int *conv_neighbors,
    const float *norm,
    float *grad_in,
    float *grad_weights_transpose) {
  int d1 = d + 1;
  int n_neighbors = static_cast<int>(pow(n + 1, d1)) - static_cast<int>(pow(n, d1));
  float *lattices_grad_out, *lattices_grad_in;

  CUDA_CHECK_ERROR(cudaMalloc((void **) (&lattices_grad_in), b * df_in * n_filled * sizeof(float)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) (&lattices_grad_out), b * df_out * n_filled * sizeof(float)));

  slice_grad_kernel << < b, opt_block_config(n_points_out, df_out) >> >
      (d, df_out, n_points_out, n_filled, grad_out, offset_out, barycentric_out, lattices_grad_out);
  CUDA_CHECK_KERNEL_ERROR();
  if (skip_conv) {
    CUDA_CHECK_ERROR(
        cudaMemcpy(lattices_grad_in,
                   lattices_grad_out,
                   b * df_in * n_filled * sizeof(float),
                   cudaMemcpyDeviceToDevice));
  } else {
//    // original convolution operation is too slow
//    conv_grad_kernel << < b, opt_block_config(n_filled, df_out) >> >
//        (df_in, df_out, n_filled, n_neighbors, lattices_grad_out, weights, conv_neighbors, lattices_grad_in);
//    CUDA_CHECK_KERNEL_ERROR();
//    weights_grad_kernel << < b, opt_block_config(n_filled, df_out) >> >
//        (df_in, df_out, n_filled, n_neighbors, lattices_grad_out, lattices_in, conv_neighbors, grad_weights);
//    CUDA_CHECK_KERNEL_ERROR();

    // conv -> matmul
    float alpha = 1.0;
    float beta = 0;
    float *col_grad;
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK_ERROR(cublasCreate(&cublas_handle));
    CUDA_CHECK_ERROR(cudaMalloc((void **) (&col_grad), df_out * n_neighbors * n_filled * sizeof(float)));
    CUDA_CHECK_KERNEL_ERROR();
    for (int i = 0; i < b; ++i) {
      CUDA_CHECK_ERROR(cudaMemset(col_grad, 0, df_out * n_neighbors * n_filled * sizeof(float)));
      img2col_grad_kernel << < df_out, opt_n_threads(n_filled) >> >
          (i, df_out, n_filled, n_neighbors, lattices_grad_out, conv_neighbors, col_grad);
      CUDA_CHECK_KERNEL_ERROR();
      CUBLAS_CHECK_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n_filled, df_in, df_out * n_neighbors,
                                     &alpha, col_grad, n_filled, weights_transpose, df_out * n_neighbors,
                                     &beta, lattices_grad_in + i * df_in * n_filled, n_filled));

      CUBLAS_CHECK_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, df_in, df_out * n_neighbors, n_filled,
                                     &alpha, lattices_in + i * df_in * n_filled, n_filled, col_grad, n_filled,
                                     &alpha, grad_weights_transpose, df_in));
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(cudaFree(col_grad));
    CUBLAS_CHECK_ERROR(cublasDestroy(cublas_handle));
  }
  splat_grad_kernel << < b, opt_block_config(n_points_in, df_in) >> >
      (d, df_in, n_points_in, n_filled, lattices_grad_in, offset_in, barycentric_in, grad_in);
  CUDA_CHECK_KERNEL_ERROR();

  CUDA_CHECK_ERROR(cudaFree(lattices_grad_in));
  CUDA_CHECK_ERROR(cudaFree(lattices_grad_out));
}
