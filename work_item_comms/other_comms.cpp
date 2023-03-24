#include <CL/sycl.hpp>
#include <array>
#include <assert.h>
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

// validate results
template<typename Array_type, typename Int_type>
void validate_plus_one(Array_type A, Int_type SIZE){
  for(int i = 0; i < SIZE; ++i){
    assert(A[i] == i + 1.0);
  }
  std::cout << "The results are correct!" << std::endl;
}

// investigating local accessors
template<typename Queue_type, typename Array_type, typename Int_type>
void local_accessors(Queue_type Q, Array_type& A, Int_type SIZE){
  // data buffer
  sycl::buffer A_buffer{A};

  Q.submit([&](sycl::handler& h){
    // global accessor
    sycl::accessor A_access{A_buffer, h};

    // 1D local accessor
    auto A_local_access = sycl::local_accessor<double, 1>(SIZE, h);

    h.parallel_for(sycl::nd_range<1>{{SIZE}, {SIZE}}, [=](sycl::nd_item<1> it){
      auto idx = it.get_global_id();
      auto local_idx = it.get_local_id();

      A_local_access[local_idx] = A_access[idx] + 1.0;
      A_access[idx] = A_local_access[local_idx];
    });
  });
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  constexpr size_t SIZE = 64;
  std::array<double, SIZE> A;

  // initializing the array
  for(int i = 0; i < SIZE; ++i){
    A[i] = i;
  }

  local_accessors(Q, A, SIZE);
  validate_plus_one(A, SIZE);

  return 0;
}