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
template<typename Scalar_type, typename Tolerance_type, typename String_type>
void check_equal(Scalar_type A, Scalar_type B, Tolerance_type tol, String_type name){
  assert(fabs(A - B) < tol);
  std::cout << "The " << name << " results are successful!" << std::endl;
}

// in order y pattern
void in_order_y_pattern(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  print_device(Q);

  double *A = sycl::malloc_shared<double>(SIZE, Q);
  double *B = sycl::malloc_shared<double>(SIZE, Q);

  Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] = i;
  });

  Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    B[i] = 2*i;
  });

  Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] += B[i];
  });

  Q.single_task([=](){
    for(int i = 1; i < SIZE; ++i){
      A[0] += A[i];
    }
  });

  Q.wait();

  const double result = (SIZE)*(SIZE-1.0)*1.5;

  check_equal(A[0], result, tol, "In Order Y Pattern");
  std::cout << "--------------------------------------" << std::endl;
}

// events y pattern
void events_y_pattern(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  double *A = sycl::malloc_shared<double>(SIZE, Q);
  double *B = sycl::malloc_shared<double>(SIZE, Q);

  auto event_1 = Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] = i;
  });

  auto event_2 = Q.parallel_for(SIZE, [=](sycl::id<1> idx){
    const int i = idx[0];
    B[i] = 2*i;
  });

  auto event_3 = Q.parallel_for(sycl::range{SIZE}, {event_1, event_2}, [=](sycl::id<1> idx){
    const int i = idx[0];
    A[i] += B[i];
  });

  Q.single_task(event_3, [=](){
    for(int i = 1; i < SIZE; ++i){
      A[0] += A[i];
    }
  });

  Q.wait();

  const double result = (SIZE)*(SIZE-1.0)*1.5;

  check_equal(A[0], result, tol, "Events Y Pattern");
  std::cout << "--------------------------------------" << std::endl;
}

// buffers y pattern
void buffers_y_pattern(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  sycl::buffer<double> A_buffer{sycl::range{SIZE}};
  sycl::buffer<double> B_buffer{sycl::range{SIZE}};

  Q.submit([&](sycl::handler &h){
    sycl::accessor A_access{A_buffer, h, sycl::read_write};
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      A_access[i] = i;
    });
  });

  Q.submit([&](sycl::handler &h){
    sycl::accessor B_access{B_buffer, h, sycl::read_write};
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      B_access[i] = 2.0*i;
    });
  });

  Q.submit([&](sycl::handler &h){
    sycl::accessor A_access{A_buffer, h, sycl::read_write};
    sycl::accessor B_access{B_buffer, h, sycl::read_only};
    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      A_access[i] += B_access[i];
    });
  });

  Q.submit([&](sycl::handler &h){
    sycl::accessor A_access{A_buffer, h, sycl::read_write};
    h.single_task([=](){
      for(int i = 0; i < SIZE; ++i){
        A_access[0] += A_access[i];
      }
    });
  });

  Q.wait();

  const double result = (SIZE)*(SIZE-1.0)*1.5;

  sycl::host_accessor A_host{A_buffer, sycl::read_only};

  check_equal(A_host[0], result, tol, "Buffers Y Pattern");
  std::cout << "--------------------------------------" << std::endl;
}

int main(){
  in_order_y_pattern();
  events_y_pattern();
  buffers_y_pattern();

  return 0;
}