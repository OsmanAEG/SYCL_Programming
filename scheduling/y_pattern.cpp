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

// check random test function
template<typename Scalar_type, typename Tolerance_type>
void check_equal(Scalar_type A, Scalar_type B, Tolerance_type tol){
  assert(fabs(A - B) < tol);
  std::cout << "The results are successful!" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  print_device(Q);

  return 0;
}