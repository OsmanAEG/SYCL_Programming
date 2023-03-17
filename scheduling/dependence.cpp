#include <cassert>
#include <CL/sycl.hpp>

constexpr int SIZE = 64;
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

// in order linear dependence
void in_order(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  print_device(Q);

  double *A = sycl::malloc_shared<double>(SIZE, Q);

  Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
  })
}

int main(){
  in_order();
  return 0;
}