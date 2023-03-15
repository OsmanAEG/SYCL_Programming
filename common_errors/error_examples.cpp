#include <CL/sycl.hpp>
#include <iostream>

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// task graph executes asynchronously from host program
template<typename Queue_type, typename Int_type>
void asynchronous_task_graph(Queue_type Q, Int_type SIZE){
  sycl::buffer<double> buf{sycl::range{SIZE}};

  Q.submit([&](sycl::handler& h){
    sycl::accessor device_acc{buf, h};
    h.parallel_for(SIZE, [=](auto& idx){
      device_acc[idx] = idx;
    });
  });

  sycl::host_accessor host_acc{buf};

  for(int i = 0; i < SIZE; ++i){
    std::cout << "Array[" << i << "] = " << host_acc[i] << "\n";
  }
}

// synchronous error caused from a larger sub buffer
template<typename Queue_type, typename Int_type>
void larger_sub_buffer(Queue_type Q, Int_type SIZE){
  sycl::buffer<double> buf_works{sycl::range{SIZE}};
  sycl::buffer<double> sub_buf_works(buf_works, sycl::id{4}, sycl::range{SIZE/4});

  std::cout << "The sub buffer worked!" << std::endl;

  sycl::buffer<double> buf_fails{sycl::range{SIZE}};
  sycl::buffer<double> sub_buf_fails(buf_fails, sycl::id{4}, sycl::range{SIZE});

  std::cout << "The sub buffer worked!" << std::endl;
}

// asynchronous error from empty command group
void empty_command_group(){
  auto asynchronous_error_handler = [](sycl::exception_list e_list){
    for(auto &e : e_list){
      try{
        std::rethrow_exception(e);
      } catch(sycl::exception& e){
        std::cout << "Asynchronous Exception!" << std::endl;
        std::cout << e.what() << std::endl;
      }
    }

    std::terminate();
  };

  sycl::queue Q1{sycl::gpu_selector_v, asynchronous_error_handler};
  sycl::queue Q2{sycl::default_selector_v, asynchronous_error_handler};

  print_device(Q1);
  print_device(Q2);

  try{
    Q1.submit([&](sycl::handler &h){
      // empty command group
    }, Q2);
  } catch(...){}
}

// unhandled exception
class random_error{};
void throw_random_error(){
  throw(random_error{});
}

// std terminate is the result of an unhandled sycl exception
void std_terminate(){
  std::terminate();
}

// catching sycl exception
template<typename Int_type>
auto catching_sycl_exception(Int_type SIZE){
  sycl::buffer<double> buf{sycl::range{SIZE}};
  try{
    sycl::buffer<double> buf_fails{sycl::range{SIZE}};
    sycl::buffer<double> sub_buf_fails(buf_fails, sycl::id{4}, sycl::range{SIZE});
  } catch(sycl::exception &e){
    std::cout << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

// catching general exceptions
template<typename Int_type>
auto catching_general_exception(Int_type SIZE){
  sycl::buffer<double> buf{sycl::range{SIZE}};
  try{
    sycl::buffer<double> buf_fails{sycl::range{SIZE}};
    sycl::buffer<double> sub_buf_fails(buf_fails, sycl::id{4}, sycl::range{SIZE});
  } catch(sycl::exception &e){
    std::cout << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  } catch(std::exception &e){
    std::cout << "std Exception: " << e.what() << std::endl;
    return 2;
  } catch(...){
    std::cout << "Unknown Exception: " << std::endl;
    return 3;
  }
  return 0;
}


int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // problem size
  constexpr size_t SIZE = 128;

  // testing error examples
  // asynchronous_task_graph(Q, SIZE);
  // larger_sub_buffer(Q, SIZE);
  // empty_command_group();
  // throw_random_error();
  // std_terminate();
  // catching_sycl_exception(SIZE);
  catching_general_exception(SIZE);
  return 0;
}