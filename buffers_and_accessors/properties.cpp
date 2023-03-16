#include <CL/sycl.hpp>
#include <mutex>

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

  constexpr int N = 64;

  double A[N];

  // creating a buffer
  sycl::buffer A_buffer{A, sycl::range(N), {sycl::property::buffer::use_host_ptr{}}};

  // creating a buffer using mutex
  std::mutex mtx;
  sycl::buffer A_mtx_buffer{A, sycl::range(N), {sycl::property::buffer::use_mutex{mtx}}};

  auto mtx_ptr = A_mtx_buffer.get_property<sycl::property::buffer::use_mutex>().get_mutex_ptr();
  std::lock_guard<std::mutex> guard{*mtx_ptr};

  // context-bound buffer
  sycl::buffer A_ctx_buffer{A, sycl::range(N),
                           {sycl::property::buffer::context_bound{Q.get_context()}}};

  return 0;
}