#include <CL/sycl.hpp>

namespace dinfo = sycl::info::device;

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

// random test function
template<typename Array_type, typename Int_type>
void equal_to_size(Array_type A, Int_type i){
  A[i] = SIZE;
}

// check random test function
template<typename Array_type>
void check_equal_to_size(Array_type A){
  for(int i = 0; i < SIZE; ++i){
    assert(fabs(A[i] - SIZE) < tol);
  }
}

// usm querries
template<typename Queue_type>
void usm_querries(Queue_type Q){
  // getting device information
  auto device  = Q.get_device();
  auto context = Q.get_context();

  bool usm_shared = device.template get_info<dinfo::usm_shared_allocations>();
  bool usm_device = device.template get_info<dinfo::usm_device_allocations>();

  // checking if we can use usm
  bool use_usm = usm_shared || usm_device;

  if(use_usm){
    double *A;
    if(usm_shared){
      A = sycl::malloc_shared<double>(SIZE, Q);
    }
    else{
      A = sycl::malloc_device<double>(SIZE, Q);
    }

    std::cout << "USM: " << ((sycl::get_pointer_type(A, context) == sycl::usm::alloc::shared)
                            ? "Shared" : "Device") << "- With Allocations on "
                         << sycl::get_pointer_device(A, context).template get_info<dinfo::name>()
                         << std::endl;

    Q.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      equal_to_size(A, i);
    });
    Q.wait();

    check_equal_to_size(A);

    sycl::free(A, Q);
  }
  else{
    sycl::buffer<double, 1> A{sycl::range{SIZE}};
    Q.submit([&](sycl::handler &h){
      sycl::accessor acc(A, h);
      h.parallel_for(SIZE, [=](sycl::id<1> idx){
        const int i = idx[0];
        equal_to_size(acc, i);
      });
    });
    Q.wait();

    sycl::host_accessor A_host{A};
    check_equal_to_size(A_host);
  }

  std::cout << "Results are Successful!" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  usm_querries(Q);

  return 0;
}

