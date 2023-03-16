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

  // passed in allocator - M buffer of doubles
  std::allocator<double> double_allocator;
  sycl::buffer<double, 1, std::allocator<double>> D{sycl::range{20}, double_allocator};

  // buffer of doubles
  double num_list[5] = {1.2, 2.4, 3.6, 4.8, 5.0};
  sycl::buffer E{num_list, sycl::range{5}};

  // double shared pointer buffer
  auto shared_pointer = std::make_shared<double>(N);
  sycl::buffer F{shared_pointer, sycl::range{1}};

  // buffer of doubles from an input iterator
  std::vector<double> double_vector;
  sycl::buffer G{double_vector.begin(), double_vector.end()};
  sycl::buffer H{double_vector};

  // buffer and non-overlapping sub-buffers
  sycl::buffer<double, 2> I{sycl::range{2, M}};
  sycl::buffer I1{I, sycl::id{0, 0}, sycl::range{1, M}};
  sycl::buffer I2{I, sycl::id{1, 0}, sycl::range{1, M}};

  return 0;
}