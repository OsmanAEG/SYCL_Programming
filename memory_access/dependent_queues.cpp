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
  std::array<int, SIZE> Arr1_host;
  std::array<int, SIZE> Arr2_host;
  std::array<int, SIZE> Arr3_host;

  // initializing host array
  for(int i = 0; i < SIZE; ++i){
    Arr1_host[i] = i;
    Arr2_host[i] = i*i;
    Arr3_host[i] = 0.0;
  }

  // initializing devide buffer
  sycl::buffer Arr1_buffer(Arr1_host);
  sycl::buffer Arr2_buffer(Arr2_host);
  sycl::buffer Arr3_buffer(Arr3_host);

  // incrementing array 1 values on device by 1
  auto event1 = Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr1_accessor[idx]++;
    });
  });

  event1.wait();

  // subtracting array 2 values on device by 1
  auto event2 = Q.submit([&](sycl::handler &h){
    sycl::accessor Arr2_accessor(Arr2_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr2_accessor[idx] -= 1.0;
    });
  });

  // adding array 1 and array 2
  auto event3 = Q.submit([&](sycl::handler &h){
    h.depends_on(event2);
    sycl::accessor Arr1_accessor(Arr1_buffer, h);
    sycl::accessor Arr2_accessor(Arr2_buffer, h);
    sycl::accessor Arr3_accessor(Arr3_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr3_accessor[idx] = Arr1_accessor[idx] + Arr2_accessor[idx];
    });
  });

  // increment array 1 twice
  auto event4 = Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr1_accessor[idx] += 2.0;
    });
  });

  // add the new array 1 to the sum
  auto event5 = Q.submit([&](sycl::handler &h){
    h.depends_on({event3, event4});
    sycl::accessor Arr1_accessor(Arr1_buffer, h);
    sycl::accessor Arr3_accessor(Arr3_buffer, h);

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr3_accessor[idx] += Arr1_accessor[idx];
    });
  });

  Q.wait();

  sycl::host_accessor Arr3_host_accessor(Arr3_buffer);

  // checking results
  for(int i = 0; i < SIZE; ++i){
    assert(Arr3_host_accessor[i] == (i+1) + (i-1) + (i+2));
  }

  std::cout << "The results are correct!" << std::endl;

  return 0;
}