#include <CL/sycl.hpp>
#include <chrono>

extern const int size_M = 256;
extern const int size_N = 128;
extern const int size_K = 512;

static const int iterations = 10;

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// basic matrix multiply


int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  return 0;
}