#include <CL/sycl.hpp>
#include <iostream>

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// task graph executes asynchronously from host program
template<typename Queue_type, typename Int_type>
void asynchronous_task_graph(Queue_type Q, Int_type SIZE){
  sycl::buffer<double> buf{sycl::range{SIZE}};

  Q.submit([&](sycl::handler& h){
    sycl::accessor device_acc{buf, h};
    h.parallel_for(SIZE, [=](auto& idx){
      device_acc[idx] = idx;
    });
  });

  sycl::host_accessor host_acc{buf};

  for(int i = 0; i < SIZE; ++i){
    std::cout << "Array[" << i << "] = " << host_acc[i] << "\n";
  }
}

// synchronous error caused from a larger sub buffer

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // problem size
  constexpr size_t SIZE = 128;

  // testing error examples
  asynchronous_task_graph(Q, SIZE);

  return 0;
}