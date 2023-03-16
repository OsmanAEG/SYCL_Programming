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

// first example
template<typename Queue_type>
void first_example(Queue_type Q){
  // creating buffers
  sycl::buffer<double> input_1{sycl::range{SIZE}};
  sycl::buffer<double> input_2{sycl::range{SIZE}};
  sycl::buffer<double> output{sycl::range{SIZE}};

  // initializing
  Q.submit([&](sycl::handler& h){
    sycl::accessor access_input_1{input_1, h};
    sycl::accessor access_input_2{input_2, h};
    sycl::accessor access_output{output, h};

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      access_input_1[i] = 5.5;
      access_input_2[i] = 8.4;
      access_output[i]  = 0.0;
    });
  });

  // vector addition
  Q.submit([&](sycl::handler& h){
    sycl::accessor access_input_1{input_1, h};
    sycl::accessor access_input_2{input_2, h};
    sycl::accessor access_output{output, h};

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      access_output[i] = access_input_1[i] + access_input_2[i];
    });
  });

  sycl::host_accessor host_output{output};

  for(int i = 0; i < SIZE; ++i){
    assert(fabs(host_output[i] - 13.9) < tol);
  }

  std::cout << "The first example vector addition results are correct!" << std::endl;
}

// second example
template<typename Queue_type>
void second_example(Queue_type Q){
  // creating buffers
  sycl::buffer<double> input_1{sycl::range{SIZE}};
  sycl::buffer<double> input_2{sycl::range{SIZE}};
  sycl::buffer<double> output{sycl::range{SIZE}};

  // initializing
  Q.submit([&](sycl::handler& h){
    sycl::accessor access_input_1{input_1, h, sycl::write_only, sycl::no_init};
    sycl::accessor access_input_2{input_2, h, sycl::write_only, sycl::no_init};
    sycl::accessor access_output{output, h, sycl::write_only, sycl::no_init};

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      access_input_1[i] = 5.5;
      access_input_2[i] = 8.4;
      access_output[i]  = 0.0;
    });
  });

  // vector addition
  Q.submit([&](sycl::handler& h){
    sycl::accessor access_input_1{input_1, h, sycl::read_only};
    sycl::accessor access_input_2{input_2, h, sycl::read_only};
    sycl::accessor access_output{output, h, sycl::read_write};

    h.parallel_for(SIZE, [=](sycl::id<1> idx){
      const int i = idx[0];
      access_output[i] = access_input_1[i] + access_input_2[i];
    });
  });

  sycl::host_accessor host_output{output};

  for(int i = 0; i < SIZE; ++i){
    assert(fabs(host_output[i] - 13.9) < tol);
  }

  std::cout << "The second example vector addition results are correct!" << std::endl;
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // examples
  first_example(Q);
  second_example(Q);

  return 0;
}