#include "combinatorial.h"
#include <cinttypes>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// #include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
using std::vector; 

auto rank_combs(py::array_t< uint16_t > combs, const int n, const int k, bool colex = true) -> py::array_t< uint64_t > {
  py::buffer_info buffer = combs.request();
  uint16_t* p = static_cast< uint16_t* >(buffer.ptr);
  const size_t N = buffer.size;
  vector< uint64_t > ranks; 
  ranks.reserve(static_cast< uint64_t >(N/k));
  auto out = std::back_inserter(ranks);
	if (colex) {
		combinatorial::rank_lex(p, p+N, size_t(n), size_t(k), out);
	} else {
		combinatorial::rank_colex(p, p+N, size_t(n), size_t(k), out);
	}
  return py::cast(ranks); 
}

// auto unrank_combs(py::array_t< int > ranks, const int n, const int k) -> py::array_t< int > {
//   py::buffer_info buffer = ranks.request();
//   int* r = static_cast< int* >(buffer.ptr);
//   const size_t N = buffer.size;
//   vector< int > simplices; 
//   simplices.reserve(static_cast< int >(N*k));
//   auto out = std::back_inserter(simplices);
//   combinatorial::unrank_lex(r, r+N, size_t(n), size_t(k), out);
//   return py::cast(simplices);
// }

// auto boundary_ranks(const int p_rank, const int n, const int k) -> py::array_t< int > {
//   vector< int > face_ranks = vector< int >();
// 	combinatorial::apply_boundary(p_rank, n, k, [&face_ranks](size_t r){
//     face_ranks.push_back(r);
//   });
//   return py::cast(face_ranks);
// }

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/combinatorial.cpp -std=c++20 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_combinatorial, m) {
  m.doc() = "Combinatorial module";
  m.def("rank_combs", &rank_combs);
  // m.def("unrank_combs", &unrank_combs);
  // m.def("boundary_ranks", &boundary_ranks);
  // m.def("interval_cost", &pairwise_cost);
  // m.def("vectorized_func", py::vectorize(my_func));s
  //m.def("call_go", &call_go);
}