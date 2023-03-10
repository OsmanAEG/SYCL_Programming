#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <assert.h>
#include <random>
#include <algorithm>

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                    Scalar_type* C, size_t M, size_t N, size_t K){
  Q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::range{M, K}, [=](sycl::id<2> idx){
      int i = idx[0];
      int j = idx[1];

      Scalar_type c_ij = 0.0;

      for(int p = 0; p < N; ++p){
        c_ij += A[i*N + p]*B[p*K + j];
      }
      C[i*K + j] = c_ij;
    });
  }).wait();
}

// nd-range parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void nd_range_parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                             Scalar_type* C, size_t M, size_t N, size_t K,
                                             size_t b){
  Q.submit([&](sycl::handler &h){
    // global nd range problem size
    sycl::range global{M, K};

    // local workgroup size
    sycl::range local{b, b};

    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it){
      int i = it.get_global_id(0);
      int j = it.get_global_id(1);

      Scalar_type c_ij = 0.0;

      for(int p = 0; p < N; ++p){
        c_ij += A[i*N + p]*B[p*K + j];
      }
      C[i*K + j] = c_ij;
    });
  }).wait();
}

// hierarchical parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void hierarchical_parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                                 Scalar_type* C, size_t M, size_t N, size_t K,
                                                 size_t b){
  Q.submit([&](sycl::handler &h){
    // number of groups
    sycl::range num_groups{M/b, K/b};

    // group size
    sycl::range group_size{b, b};

    h.parallel_for_work_group(num_groups, group_size, [=](sycl::group<2> grp){
      int ib = grp.get_group_id(0);
      int jb = grp.get_group_id(1);

      grp.parallel_for_work_item([&](sycl::h_item<2> it){
        int i = ib*b + it.get_local_id(0);
        int j = jb*b + it.get_local_id(1);

        Scalar_type c_ij = 0.0;

        for(int p = 0; p < N; ++p){
          c_ij += A[i*N + p]*B[p*K + j];
        }

        C[i*K + j] = c_ij;
      });
    });
  }).wait();
}

// logical hierarchical parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void logical_hierarchical_parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                                         Scalar_type* C, size_t M, size_t N, size_t K,
                                                         size_t b){
  Q.submit([&](sycl::handler &h){
    // number of groups
    sycl::range num_groups{M/b, K/b};

    // group size
    sycl::range group_size{b, b};

    h.parallel_for_work_group(num_groups, [=](sycl::group<2> grp){
      int ib = grp.get_group_id(0);
      int jb = grp.get_group_id(1);

      grp.parallel_for_work_item(group_size, [&](sycl::h_item<2> it){
        int i = ib*b + it.get_logical_local_id(0);
        int j = jb*b + it.get_logical_local_id(1);

        Scalar_type c_ij = 0.0;

        for(int p = 0; p < N; ++p){
          c_ij += A[i*N + p]*B[p*K + j];
        }

        C[i*K + j] = c_ij;
      });
    });
  }).wait();
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // matrix dimensional value
  constexpr size_t M = 512;
  constexpr size_t N = 512;
  constexpr size_t K = 512;

  // local work group size
  constexpr size_t b = 4;

  // tolerance
  const double tol = 1.0E-6;

  // matrices on host memory
  std::vector<double> A_host(M*N);
  std::vector<double> B_host(N*K);
  std::vector<double> C_host(M*K);

  // creating a random distribution
  std::default_random_engine generate(53);
  std::uniform_real_distribution<double> distribution(0.0, 2.0);

  auto random_number_generator = [&](){
    return distribution(generate);
  };

  // filling the input and output matrices on host
  std::generate(A_host.begin(), A_host.end(), random_number_generator);
  std::generate(B_host.begin(), B_host.end(), random_number_generator);
  std::fill(C_host.begin(), C_host.end(), 0.0);

  // allocating device memory
  double *A_device = sycl::malloc_device<double>(M*N, Q);
  double *B_device = sycl::malloc_device<double>(N*K, Q);
  double *C_device = sycl::malloc_device<double>(M*K, Q);

  // copying host to device memory
  Q.memcpy(A_device, &A_host[0], M*N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], N*K*sizeof(double));
  Q.memcpy(C_device, &C_host[0], M*K*sizeof(double));

  //parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K);
  //nd_range_parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K, b);
  //hierarchical_parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K, b);
  logical_hierarchical_parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K, b);

  // copying device to host memory
  Q.memcpy(&C_host[0], C_device, M*K*sizeof(double));

  // confirming results
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < K; ++j){
      double c_ij = 0.0;
      for(int p = 0; p < N; ++p){
        c_ij += A_host[i*N + p]*B_host[p*K + j];
      }
      assert(std::fabs(C_host[i*K + j] - c_ij) < tol);
    }
  }

  std::cout << "The parallel matrix multiplication was successful!" << std::endl;

  return 0;
}