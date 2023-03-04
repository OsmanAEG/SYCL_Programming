#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <assert.h>

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// parallel vector addition
template<typename Queue_type, typename Scalar_type>
void parallel_matrix_addition(Queue_type Q, Scalar_type* A, Scalar_type* B,
                              Scalar_type* C, size_t M, size_t N){
  Q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::range{M, N}, [=](sycl::id<2> idx){
      int i = idx[0];
      int j = idx[1];
      C[i + M*j] = A[i + M*j] + B[i + M*j];
    });
  }).wait();
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // vector dimensional value
  constexpr size_t M = 512;
  constexpr size_t N = 512;

  // tolerance
  const double tol = 1.0E-6;

  // vectors on host memory
  std::vector<double> A_host(M*N);
  std::vector<double> B_host(M*N);
  std::vector<double> C_host(M*N);

  // filling the input and output vectors on host
  std::fill(A_host.begin(), A_host.end(), 8.39);
  std::fill(B_host.begin(), B_host.end(), 2.67);
  std::fill(C_host.begin(), C_host.end(), 0.00);

  // allocating device memory
  double *A_device = sycl::malloc_device<double>(M*N, Q);
  double *B_device = sycl::malloc_device<double>(M*N, Q);
  double *C_device = sycl::malloc_device<double>(M*N, Q);

  // copying host to device memory
  Q.memcpy(A_device, &A_host[0], M*N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], M*N*sizeof(double));
  Q.memcpy(C_device, &C_host[0], M*N*sizeof(double));

  parallel_matrix_addition(Q, A_device, B_device, C_device, M, N);

  // copying device to host memory
  Q.memcpy(&A_host[0], A_device, M*N*sizeof(double));
  Q.memcpy(&B_host[0], B_device, M*N*sizeof(double));
  Q.memcpy(&C_host[0], C_device, M*N*sizeof(double));

  // confirming results
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < M; ++j){
      assert(C_host[i + M*j] == A_host[i + M*j] + B_host[i + M*j]);
    }
  }

  std::cout << "The parallel matrix addition was successful!" << std::endl;

  return 0;
}