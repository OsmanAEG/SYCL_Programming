#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <assert.h>

constexpr int SIZE = 256;

// prints device name
template<typename Queue_type, typename String_type>
void print_device(Queue_type& Q, String_type name){
  std::cout << name << std::endl;
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

  // allocating memory on device
  int *Arr_device = sycl::malloc_device<int>(SIZE, Q);

  // filling the host array with index values
  for(int i = 0; i < SIZE; ++i){
    Arr_host[i] = i;
  }

  // copying host to device
  Q.submit([&](sycl::handler &h){
    h.memcpy(Arr_device, &Arr_host[0], SIZE*sizeof(int));
  });

  Q.wait();

  // incrementing array values on device
  Q.submit([&](sycl::handler &h){
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr_device[idx]++;
    });
  });

  Q.wait();

  // copying device to host
  Q.submit([&](sycl::handler &h){
    h.memcpy(&Arr_host[0], Arr_device, SIZE*sizeof(int));
  });

  Q.wait();

  // checking results
  for(int i = 0; i < SIZE; ++i){
    assert(Arr_host[i] = i + 1);
  }

  std::cout << "The results are correct!" << std::endl;

  sycl::free(Arr_device, Q);
  return 0;
}