#include <CL/sycl.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <assert.h>
#include <random>
#include <algorithm>

extern const size_t M = 256;
extern const size_t N = 128;
extern const size_t K = 512;

static const int attempts = 10;

// matrix multiplication selection type
static const int selection = 0;

// prints device name
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

// verifying matrix multiply results
template<typename Scalar_type>
void check_matrix_multiply(std::vector<Scalar_type> A,
                           std::vector<Scalar_type> B,
                           std::vector<Scalar_type> C,
                           Scalar_type tol){
  // confirming results
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      double c_ij = 0.0;
      for(int k = 0; k < K; ++k){
        c_ij += A[i*K + k]*B[k*N + j];
      }
      assert(std::fabs(C[i*N + j] - c_ij) < tol);
    }
  }

  std::cout << "The matrix multiply results are correct!" << std::endl;
}

// basic matrix multiply
template<typename Queue_type, typename Scalar_type>
void basic_matrix_multiply(Queue_type Q, std::vector<Scalar_type>& A,
                                         std::vector<Scalar_type>& B,
                                         std::vector<Scalar_type>& C){
  sycl::buffer<Scalar_type, 2> A_buffer{A.data(), sycl::range<2>{M, K}};
  sycl::buffer<Scalar_type, 2> B_buffer{B.data(), sycl::range<2>{K, N}};
  sycl::buffer<Scalar_type, 2> C_buffer{C.data(), sycl::range<2>{M, N}};

  Q.submit([&](sycl::handler& h){
    sycl::accessor A_access{A_buffer, h, sycl::read_only};
    sycl::accessor B_access{B_buffer, h, sycl::read_only};
    sycl::accessor C_access{C_buffer, h, sycl::read_write};

    h.parallel_for(sycl::range{M, N}, [=](sycl::id<2> idx){
      const int i = idx[0];
      const int j = idx[1];

      Scalar_type c_ij = 0;

      for(int k = 0; k < K; ++k){
        c_ij += A_access[i][k] * B_access[k][j];
      }

      C_access[i][j] = c_ij;
    });
  });

  Q.wait();
}

// benchmark time
template<typename Queue_type, typename Scalar_type>
void time_bench(Queue_type Q, std::vector<Scalar_type>& A,
                              std::vector<Scalar_type>& B,
                              std::vector<Scalar_type>& C){
  // clock
  using ns = std::chrono::nanoseconds;
  ns::rep min_time = std::numeric_limits<ns::rep>::max();

  // running the test
  for(int i = 0; i < attempts; ++i){
    auto start_time = std::chrono::steady_clock::now();
    auto interval = std::chrono::steady_clock::now() - start_time;

    if constexpr (selection == 0){
      basic_matrix_multiply(Q, A, B, C);
    }

    auto time = std::chrono::duration_cast<ns>(interval).count();
    min_time = std::min(time, min_time);
  }
  std::cout << "The minimum execution time is " << min_time << " ns!" << std::endl;
}

template<typename Queue_type, typename Scalar_type>
void unit_test(Queue_type Q, std::vector<Scalar_type>& A,
                              std::vector<Scalar_type>& B,
                              std::vector<Scalar_type>& C){

  if constexpr (selection == 0){
    basic_matrix_multiply(Q, A, B, C);
  }
}

int main(){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};
  print_device(Q);

  // tolerance value
  const double tol = 1.0E-6;

  // creating input and output vectors
  std::vector<double> A(M*K);
  std::vector<double> B(K*N);
  std::vector<double> C(M*N);

  // creating a random distribution
  std::default_random_engine generate(68);
  std::uniform_real_distribution<double> distribution(0.0, 2.0);

  auto random_number_generator = [&](){
    return distribution(generate);
  };

  // filling the input and output matrices on host
  std::generate(A.begin(), A.end(), random_number_generator);
  std::generate(B.begin(), B.end(), random_number_generator);
  std::fill(C.begin(), C.end(), 0.0);

  // matrix multiply test
  unit_test(Q, A, B, C);

  // timed benchmark matrix multiply
  //time_bench(Q, A, B, C);

  // validating the results of the matrix multiply
  check_matrix_multiply(A, B, C, tol);

  return 0;
}