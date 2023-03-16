#include <CL/sycl.hpp>

int main(){
  const int M = 16;
  const int N = 64;

  // default allocation - M x N buffer of doubles
  sycl::buffer<double, 2, sycl::buffer_allocator<double>> A{sycl::range<2>{M, N}};

  // CTAD range - M x N buffer of doubles
  sycl::buffer<double, 2> B{sycl::range{M, N}};

  // default std allocator - M buffer of doubles
  sycl::buffer<double, 1, std::allocator<double>> C{sycl::range{20}};

  return 0;
}