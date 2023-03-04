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

template<typename Queue_type>
void read_write_test_1(Queue_type Q){
  // arrays on host memory
  std::array<int, SIZE> Arr1_host;
  std::array<int, SIZE> Arr2_host;
  std::array<int, SIZE> Arr3_host;

  // initializing host arrays
  for(int i = 0; i < SIZE; ++i){
    Arr1_host[i] = i;
    Arr2_host[i] = 0;
    Arr3_host[i] = i*i*i;
  }

  // establishing memory buffers
  sycl::buffer Arr1_buffer{Arr1_host};
  sycl::buffer Arr2_buffer{Arr2_host};
  sycl::buffer Arr3_buffer{Arr3_host};

  // first kernel execution
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h, sycl::read_only);
    sycl::accessor Arr2_accessor(Arr2_buffer, h, sycl::write_only);
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr2_accessor[idx] = Arr1_accessor[idx]*Arr1_accessor[idx];
    });
  }).wait();

  // second kernel execution
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h, sycl::read_only);
    sycl::accessor Arr2_accessor(Arr2_buffer, h, sycl::read_only);
    sycl::accessor Arr3_accessor(Arr3_buffer, h, sycl::read_write);
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr3_accessor[idx] += Arr1_accessor[idx] + Arr2_accessor[idx];
    });
  }).wait();

  // reading memory on host
  sycl::host_accessor Arr1_host_accessor(Arr1_buffer, sycl::read_only);
  sycl::host_accessor Arr2_host_accessor(Arr2_buffer, sycl::read_only);
  sycl::host_accessor Arr3_host_accessor(Arr3_buffer, sycl::read_only);

  for(int i = 0; i < SIZE; ++i){
    assert(Arr1_host_accessor[i] == i);
    assert(Arr2_host_accessor[i] == i*i);
    assert(Arr3_host_accessor[i] == i*i*i + i + i*i);
  }

  std::cout << "Test 1 results passed!" << std::endl;
}

template<typename Queue_type>
void read_write_test_2(Queue_type Q){
  // arrays on host memory
  std::array<int, SIZE> Arr1_host;
  std::array<int, SIZE> Arr2_host;
  std::array<int, SIZE> Arr3_host;

  // initializing host arrays
  for(int i = 0; i < SIZE; ++i){
    Arr1_host[i] = i;
    Arr2_host[i] = 0;
    Arr3_host[i] = i*i*i;
  }

  // establishing memory buffers
  sycl::buffer Arr1_buffer{Arr1_host};
  sycl::buffer Arr2_buffer{Arr2_host};
  sycl::buffer Arr3_buffer{Arr3_host};

  // first kernel execution
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h, sycl::read_only);
    sycl::accessor Arr2_accessor(Arr2_buffer, h, sycl::write_only);
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr2_accessor[idx] = Arr1_accessor[idx]*Arr1_accessor[idx];
    });
  }).wait();

  // second kernel execution
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr1_accessor(Arr1_buffer, h, sycl::read_only);
    sycl::accessor Arr2_accessor(Arr2_buffer, h, sycl::read_only);
    sycl::accessor Arr3_accessor(Arr3_buffer, h, sycl::read_write);
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr3_accessor[idx] += Arr1_accessor[idx] + Arr2_accessor[idx];
    });
  }).wait();

  // second kernel execution
  Q.submit([&](sycl::handler &h){
    sycl::accessor Arr2_accessor(Arr2_buffer, h, sycl::write_only);
    sycl::accessor Arr3_accessor(Arr3_buffer, h, sycl::write_only);
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      Arr2_accessor[idx] = 2.0;
      Arr3_accessor[idx] = 3.0;
    });
  }).wait();

  // reading memory on host
  sycl::host_accessor Arr1_host_accessor(Arr1_buffer, sycl::read_only);
  sycl::host_accessor Arr2_host_accessor(Arr2_buffer, sycl::read_only);
  sycl::host_accessor Arr3_host_accessor(Arr3_buffer, sycl::read_only);

  for(int i = 0; i < SIZE; ++i){
    assert(Arr1_host_accessor[i] == i);
    assert(Arr2_host_accessor[i] == 2.0);
    assert(Arr3_host_accessor[i] == 3.0);
  }

  std::cout << "Test 2 results passed!" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // running the tests
  read_write_test_1(Q);
  read_write_test_2(Q);

  return 0;
}