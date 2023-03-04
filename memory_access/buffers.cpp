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
  std::array<int, SIZE> Arr_host;

  // initializing host array
  for(int i = 0; i < SIZE; ++i){
    Arr_host[i] = i;
  }

  // initializing devide buffer
  sycl::buffer Arr_buffer(Arr_host);

  // incrementing array values on device
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr_accessor(Arr_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr_accessor[idx]++;
    });
  });

  Q.wait();

  sycl::host_accessor Arr_host_accessor(Arr_buffer);

  // checking results
  for(int i = 0; i < SIZE; ++i){
    assert(Arr_host_accessor[i] == i + 1);
  }

  std::cout << "The results are correct!" << std::endl;

  return 0;
}