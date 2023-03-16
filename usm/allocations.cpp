#include <CL/sycl.hpp>
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

  // C style memory allocation
  double *A = static_cast<double*>(sycl::malloc_shared(SIZE*sizeof(double), Q));

  // C++ style memory allocation
  double *B = sycl::malloc_shared<double>(SIZE, Q);

  // C++ allocator
  sycl::usm_allocator<double, sycl::usm::alloc::shared> alloc(Q);
  double *C = alloc.allocate(SIZE);

  // deallocate memory
  sycl::free(A, Q.get_context());
  sycl::free(B, Q);
  alloc.deallocate(C, SIZE);

  return 0;
}