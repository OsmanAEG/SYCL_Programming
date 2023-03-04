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
void parallel_vector_addition(Queue_type Q, Scalar_type* A, Scalar_type* B,
                              Scalar_type* C, size_t SIZE){
  Q.submit([&](sycl::handler &h){
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      C[idx] = A[idx] + B[idx];
    });
  }).wait();
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // vector dimensional value
  constexpr size_t SIZE = 512;

  // tolerance
  const double tol = 10.0E-6;

  // vectors on host memory
  std::vector<double> A_host(SIZE);
  std::vector<double> B_host(SIZE);
  std::vector<double> C_host(SIZE);

  // filling the input and output vectors on host
  std::fill(A_host.begin(), A_host.end(), 8.39);
  std::fill(B_host.begin(), B_host.end(), 2.67);
  std::fill(C_host.begin(), C_host.end(), 0.00);

  // allocating device memory
  double *A_device = sycl::malloc_device<double>(SIZE, Q);
  double *B_device = sycl::malloc_device<double>(SIZE, Q);
  double *C_device = sycl::malloc_device<double>(SIZE, Q);

  // copying host to device memory
  Q.memcpy(A_device, &A_host[0], SIZE*sizeof(double));
  Q.memcpy(B_device, &B_host[0], SIZE*sizeof(double));
  Q.memcpy(C_device, &C_host[0], SIZE*sizeof(double));

  parallel_vector_addition(Q, A_device, B_device, C_device, SIZE);

  // copying device to host memory
  Q.memcpy(&A_host[0], A_device, SIZE*sizeof(double));
  Q.memcpy(&B_host[0], B_device, SIZE*sizeof(double));
  Q.memcpy(&C_host[0], C_device, SIZE*sizeof(double));

  // confirming results
  for(int i = 0; i < SIZE; ++i){
    assert(C_host[i] == A_host[i] + B_host[i]);
  }

  std::cout << "The parallel vector addition was successful!" << std::endl;

  return 0;
}