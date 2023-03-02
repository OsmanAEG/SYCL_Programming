#include <CL/sycl.hpp>
#include <array>
#include <iostream>

// set array values to index values in parallel
template<typename Sycl_Queue, typename Array_type, typename Int_type>
void set_index(Sycl_Queue Q, Array_type& A, Int_type N){
  sycl::buffer A_buffer{A};
  Q.submit([&](sycl::handler& h){
    sycl::accessor A_accessor{A_buffer, h};
    h.parallel_for(N, [=](auto& idx){
      A_accessor[idx] = idx;
    });
  });
}

int main(){
  constexpr int N = 20;
  std::array<int, N> arr;

  // device queue
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  set_index(Q, arr, N);

  for(int i = 0; i < N; ++i){
    std::cout << "Array[" << i << "] = " << arr[i] << std::endl;
  }

  return 0;
}