#include <CL/sycl.hpp>
#include <iostream>

// prints device name
template<typename Queue_type, typename String_type>
void print_device(Queue_type& Q, String_type name){
  std::cout << name << std::endl;
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// selects device implicitly
void implicit_selector(){
  sycl::queue Q;
  print_device(Q, "Implicit Device Selector");
}

// selects default device
void default_selector(){
  sycl::queue Q{sycl::default_selector_v};
  print_device(Q, "Default Device Selector");
}

// selects gpu device
void gpu_selector(){
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q, "GPU Device Selector");
}

// selects multiple devices
void multiple_device_selector(){
  sycl::queue Q1;
  sycl::queue Q2{sycl::default_selector_v};
  sycl::queue Q3{sycl::gpu_selector_v};
  print_device(Q1, "Implicit Device Selector");
  print_device(Q2, "Default Device Selector");
  print_device(Q3, "GPU Device Selector");
}

// checks NVIDIA device availablity
int nvidia_device_availability(const sycl::device &D){
  if(D.get_info<sycl::info::device::name>().find("NVIDIA") != std::string::npos){
    return 1;
  }
  return -1;
}

// checks AMD device availablity
int amd_device_availability(const sycl::device &D){
  if(D.get_info<sycl::info::device::name>().find("AMD") != std::string::npos){
    return 1;
  }
  return -1;
}

// selects nvidia device based on availability
void nvidia_device_selection(){
  sycl::queue Q(nvidia_device_availability);
  print_device(Q, "NVIDIA Device Selector");
}

// selects AMD device based on availability
void amd_device_selection(){
  sycl::queue Q(amd_device_availability);
  print_device(Q, "AMD Device Selector");
}

int main(){
  //implicit_selector();
  //default_selector();
  //gpu_selector();
  //multiple_device_selector();
  nvidia_device_selection();
  //amd_device_selection();
}