#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <assert.h>

constexpr int SIZE = 256;

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // allocating memory on host
  int *Arr_host = sycl::malloc_host<int>(SIZE, Q);

  // allocating shared memory between host and device
  int *Arr_shared = sycl::malloc_shared<int>(SIZE, Q);

  // filling the host array with index values
  for(int i = 0; i < SIZE; ++i){
    Arr_host[i] = i;
  }

  // incrementing array values on device
  Q.submit([&](sycl::handler &h){
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr_shared[idx] = Arr_host[idx] + 1;
    });
  });

  Q.wait();

  // updating host array
  for(int i = 0; i < SIZE; ++i){
    Arr_host[i] = Arr_shared[i];
  }

  // checking results
  for(int i = 0; i < SIZE; ++i){
    assert(Arr_host[i] = i + 1);
  }

  std::cout << "The results are correct!" << std::endl;

  sycl::free(Arr_host, Q);
  sycl::free(Arr_shared, Q);
  return 0;
}