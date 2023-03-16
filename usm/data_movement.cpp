#include <CL/sycl.hpp>
#include <array>
#include <assert.h>

constexpr int SIZE = 256;
constexpr double tol = 1.0E-6;

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// explicit usm data movement
template<typename Queue_type>
void explicit_data_movement(Queue_type Q){
  std::array<double, SIZE> A_host;
  double* A_device = sycl::malloc_device<double>(SIZE, Q);

  // filling up the host array
  for(int i = 0; i < SIZE; ++i){
    A_host[i] = i;
  }

  // moving host to device memory
  Q.submit([&](sycl::handler& h){
    h.memcpy(A_device, &A_host[0], SIZE*sizeof(double));
  });

  Q.wait();

  // task
  Q.submit([&](sycl::handler& h){
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const auto i = idx[0];
      A_device[i] = A_device[i]*i;
    });
  });

  Q.wait();

  // moving device to host memory
  Q.submit([&](sycl::handler& h){
    h.memcpy(&A_host[0], A_device, SIZE*sizeof(double));
  });

  Q.wait();

  // checking results
  for(int i = 0; i < SIZE; ++i){
    assert(fabs(A_host[i] - i*i) < tol);
  }

  std::cout << "The explicit data movement was successful!" << std::endl;

  sycl::free(A_device, Q);
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  explicit_data_movement(Q);
}