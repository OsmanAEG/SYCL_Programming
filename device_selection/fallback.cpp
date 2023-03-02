#include <CL/sycl.hpp>
#include <array>
#include <assert.h>
#include <iostream>

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
  // dimensional values
  constexpr size_t N_global = 16;
  constexpr size_t N_local  = 16;

  // tolerance
  constexpr double tol = 10E-6;

  // buffer
  sycl::buffer<double, 2> buffer{sycl::range{N_global, N_global}};

  // selecting device queues
  sycl::queue Q_default{sycl::default_selector_v};
  sycl::queue Q_gpu{sycl::gpu_selector_v};

  // checking selected devices
  print_device(Q_default, "Default Device Selector");
  print_device(Q_gpu,     "GPU Device Selector");

  // work group range
  sycl::nd_range wg_range{sycl::range{N_global, N_global}, sycl::range{N_local, N_local}};

  // submitting work to queue with a fallback device selection
  Q_gpu.submit([&](sycl::handler& h){
    sycl::accessor device_accessor{buffer, h};

    h.parallel_for(wg_range, [=](auto idx){
      auto global_id = idx.get_global_id();
      device_accessor[global_id] = global_id[0] + global_id[1];
    });
  }, Q_default);

  sycl::host_accessor host_accessor{buffer};

  // checking results
  for(int i = 0; i < N_global; ++i){
    for(int j = 0; j < N_global; ++j){
      assert(abs(host_accessor[i][j] - (i + j)) < tol);
    }
  }

  std::cout << "The answers are correct!" << std::endl;
  return 0;
}