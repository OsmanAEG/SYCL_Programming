#include <CL/sycl.hpp>
#include <iostream>
#include <assert.h>

// number of threads per block
constexpr int number_of_threads = 64;

// number of blocks
constexpr int number_of_blocks = 2000;

// problem size
constexpr int SIZE = number_of_threads*number_of_blocks;

// result tolerance
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

// example case using prefetch
template<typename Queue_type>
void example_prefetch_case(Queue_type Q){
  double *A_shared = sycl::malloc_shared<double>(SIZE, Q);
  double *A_read_only = sycl::malloc_shared<double>(number_of_threads, Q);

  // initializing data
  for(int i = 0; i < number_of_threads; ++i){
    A_read_only[i] = i;
  }

  // marking shared data as read only (copies instead of migrate)
  int hw_specific_advice = 0;
  Q.mem_advise(A_read_only, number_of_threads, hw_specific_advice);
  auto e = Q.prefetch(A_shared, number_of_threads);

  for(int b = 0; b < number_of_blocks; ++b){
    Q.parallel_for(sycl::range{number_of_threads}, e, [=](sycl::id<1> idx){
      const int i = idx[0];
      A_shared[b*number_of_threads + i] += A_read_only[i];
    });

    if((b + 1) < number_of_blocks){
      e = Q.prefetch(A_shared + (b + 1)*number_of_threads, number_of_threads);
    }
  }
  Q.wait();

  for(int i = 0; i < number_of_blocks; ++i){
    for(int j = 0; j < number_of_threads; ++j){
      assert(A_shared[j] - j < tol);
    }
  }

  std::cout << "Prefetching was Successful!" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  example_prefetch_case(Q);
}