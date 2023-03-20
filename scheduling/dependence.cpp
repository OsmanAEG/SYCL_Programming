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

// in order linear dependence
template<typename Queue_type>
void in_order_linear_dependance(Queue_type Q){
  double *A = sycl::malloc_shared<double>(SIZE, Q);

  Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] = i;
  });

  Q.single_task([=](){
    for(int i = 1; i < SIZE; ++i){
      A[0] += A[i];
    }
  });

  Q.wait();

  const double result = (SIZE)*(SIZE-1.0)*0.5;

  check_equal(A[0], result, 1.0E-6);
}

// event linear dependance
template<typename Queue_type>
void event_linear_dependance(Queue_type Q){
  double *A = sycl::malloc_shared<double>(SIZE, Q);

  auto event = Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] = i;
  });

  Q.submit([&](sycl::handler &h){
    h.depends_on(event);
    h.single_task([=](){
      for(int i = 1; i < SIZE; ++i){
        A[0] += A[i];
      }
    });
  });

  Q.wait();

  const double result = (SIZE)*(SIZE-1.0)*0.5;

  check_equal(A[0], result, 1.0E-6);
}

// buffer in order linear dependance
template<typename Queue_type>
void buffer_in_order_linear_dependance(Queue_type Q){
  sycl::buffer<double> A_buffer{sycl::range{SIZE}};

  Q.submit([&](sycl::handler &h){
    sycl::accessor A_accessor{A_buffer, h};
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      A_accessor[i] = i;
    });
  });

  Q.submit([&](sycl::handler &h){
    sycl::accessor A_accessor{A_buffer, h};
    h.single_task([=](){
      for(int i = 0; i < SIZE; ++i){
        A_accessor[0] += A_accessor[i];
      }
    });
  });

  sycl::host_accessor A_host{A_buffer};

  const double result = (SIZE)*(SIZE-1.0)*0.5;

  check_equal(A_host[0], result, 1.0E-6);
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  print_device(Q);

  // running functions
  in_order_linear_dependance(Q);
  event_linear_dependance(Q);
  buffer_in_order_linear_dependance(Q);
  return 0;
}