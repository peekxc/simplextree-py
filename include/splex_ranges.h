#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "combinatorial.h"

typedef float value_t;
typedef int64_t index_t;
using std::vector;
using std::array; 
using namespace combinatorial;

template< uint8_t dim, bool colex = true, typename index_t = uint_fast64_t > 
struct RankRange {
  static_assert(std::is_integral< index_t >::value, "Must be integral");
  const size_t n; 
  const vector< index_t > ranks;
  mutable array< uint16_t, dim + 1 > labels; // intermediate storage
  
  using iterator = typename vector< index_t >::iterator;
  using const_iterator = typename vector< index_t >::const_iterator;

  RankRange(const vector< index_t > _ranks, const size_t _n) : n(_n), ranks(_ranks){
    combinatorial::BC.precompute(_n, dim+1);
  }
  ~RankRange(){
    combinatorial::BC.BT.clear();
    combinatorial::BC.BT.shrink_to_fit();
    combinatorial::BC.pre_n = 0;
    combinatorial::BC.pre_k = 0;
  }

  struct RankLabelIterator {
    const size_t _n; 
    const_iterator _it; 
    array< uint16_t, dim + 1 >& _labels; 
    
    RankLabelIterator(const size_t __n, const_iterator it, array< uint16_t, dim + 1 >& labels) : _n(__n), _it(it), _labels(labels){};

    uint16_t* operator*() const {
      if constexpr (colex){
        unrank_colex< false >(_it, _it+1, _n, dim+1, (uint16_t*) &_labels[0]);
      } else {
        unrank_lex< true, false >(_it, _it+1, _n, dim+1, (uint16_t*) &_labels[0]); 
      }
      return &_labels[0];
    }
    void operator++() { _it++; }
    bool operator!=(RankLabelIterator o) const { return _it != o._it; }
    
    template< bool ranks = false, typename Lambda > 
    void boundary(Lambda&& f){
      if constexpr (!ranks){
        uint16_t* c_labels = this->operator*();
        combinatorial::for_each_combination(c_labels, c_labels + dim, c_labels + dim + 1, [&](auto b, [[maybe_unused]] auto e){
          f(b, e); 
          return false; 
        });
      } else if constexpr (ranks && colex){
        index_t idx_below = *_it;
        index_t idx_above = 0; 
        index_t j = _n - 1, k = dim;
        for (index_t ki = 0; ki <= dim; ++ki){
          j = combinatorial::get_max_vertex< false >(idx_below, k + 1, j);
          index_t c = combinatorial::BinomialCoefficient< false >(j, k + 1);
          index_t face_index = idx_above - c + idx_below;
          idx_below -= c;
          idx_above += combinatorial::BinomialCoefficient< false >(j, k);
          --k;
          f(face_index);      
        }
      } else {
        uint16_t* c_labels = this->operator*();
        const index_t N = combinatorial::BinomialCoefficient< false >(_n, dim); 
        for_each_combination(c_labels, c_labels + dim, c_labels + dim + 1, [&](auto b, [[maybe_unused]] auto e){
          f(combinatorial::rank_lex_k< false >(b, _n, dim, N)); 
          return false; 
        });
        // throw std::invalid_argument("Haven't implemented yet.");
      }
    }
  };

  auto begin() {
    return RankLabelIterator(n, ranks.begin(), labels);
  }
  auto end() {
    return RankLabelIterator(n, ranks.end(), labels);
  }
};


// Stores the simplex labels as-is, in the prescribed ordering
template< uint8_t dim, bool colex = true, typename index_t = uint16_t > 
struct SimplexRange {
  static constexpr bool colex_order = colex; 
  using iterator = typename vector< index_t >::iterator;
  using const_iterator = typename vector< index_t >::const_iterator;
  static_assert(std::is_integral< index_t >::value, "Must be integral"); 
  
  // Fields 
  const vector< uint16_t > s_labels;    // the actual labels
  const size_t n;                       // alphabet size
  
  SimplexRange(const vector< uint16_t > S, const size_t _n) : s_labels(S), n(_n) {
    // TODO: check labels explicitly and sort them for lex or colex 

    combinatorial::BC.precompute(n, dim+1);
  }

  ~SimplexRange(){
    if (!combinatorial::keep_table_alive){
      combinatorial::BC.BT.clear();
      combinatorial::BC.BT.shrink_to_fit();
      combinatorial::BC.pre_n = 0;
      combinatorial::BC.pre_k = 0;
    }
  }

  struct SimplexLabelIterator {
    using iterator_category = std::forward_iterator_tag;
    // using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = uint16_t;
    using pointer = uint16_t*;
    using reference = uint16_t&;

    const index_t _n; 
    const index_t _N; 
    const_iterator _it;
    SimplexLabelIterator(const_iterator it, const index_t __n) : _n(__n), _N(combinatorial::BinomialCoefficient< true >(_n, dim)), _it(it) {};
    SimplexLabelIterator& operator=(const SimplexLabelIterator& it){
      _it = it._it; 
      return *this; 
    }
    uint16_t* operator*() noexcept { return (uint16_t*) &(*_it); }
    constexpr void operator++() noexcept { _it += (dim+1); }
    constexpr void operator--() noexcept { _it -= (dim+1); }
    constexpr bool operator!=(SimplexLabelIterator o) const noexcept { return _it != o._it; }
    // operator SimplexLabelIterator() const { return SimplexLabelIterator(_it, _n); }

    // Boilerplate 
    // constexpr SimplexLabelIterator& operator++() noexcept { _it += (dim+1); return *this; }
    // constexpr SimplexLabelIterator& operator--() noexcept { _it -= (dim+1); return *this; }
    // constexpr SimplexLabelIterator operator+(const difference_type& diff) const noexcept { return SimplexLabelIterator(_it + diff*(dim+1), _n); }
    // constexpr SimplexLabelIterator operator-(const difference_type& diff) const noexcept { return SimplexLabelIterator(_it - diff*(dim+1), _n); }
    // constexpr reference operator[] (const difference_type& offset) const noexcept { return *(_it + offset*(dim+1)); }
    // constexpr difference_type operator-(const SimplexLabelIterator& it) const { return this->_it - it._it; }

    template< typename Lambda > 
    void boundary_labels(Lambda&& f){
      uint16_t* labels = this->operator*();
      combinatorial::for_each_combination(labels, labels + dim, labels + dim + 1, [&f](auto b, auto e){ 
        f(b,e);
        return false; 
      });
    }

    // boundary ranks (lambda version)
    template< typename Lambda > 
    void boundary_ranks(Lambda&& f){
      uint16_t* labels = this->operator*();
      if constexpr (dim == 1){ // labels size == 2
        f(labels[0]);
        f(labels[1]);
      } else if constexpr (dim == 2){
        if constexpr(colex){
          f(rank_colex_2(labels[0], labels[1]));
          f(rank_colex_2(labels[0], labels[2]));
          f(rank_colex_2(labels[1], labels[2]));
        } else {
          f(rank_lex_2(labels[0], labels[1], _n));
          f(rank_lex_2(labels[0], labels[2], _n));
          f(rank_lex_2(labels[1], labels[2], _n));
        }
      } else {
        for_each_combination(labels, labels + dim, labels + dim + 1, [&](auto b, [[maybe_unused]] auto e){
          f(rank_comb< colex, false >(b,_n,dim));
          return false; 
        });    
      }
    } // boundary_ranks 

    // boundary ranks (tuple version)
    auto boundary_ranks() -> tuple_of< uint64_t, dim+1 >{
      uint16_t* labels = this->operator*();
      if constexpr (dim == 1){ 
        return std::make_tuple(labels[0], labels[1]);
      } else if constexpr(dim == 2){
        if constexpr(colex){
          return std::make_tuple(rank_colex_2(labels[0], labels[1]), rank_colex_2(labels[0], labels[2]), rank_colex_2(labels[1], labels[2]));
        } else {
          return std::make_tuple(rank_lex_2(labels[0], labels[1], _n), rank_lex_2(labels[0], labels[2], _n), rank_lex_2(labels[1], labels[2], _n));
        }
      } else {
        std::array< uint64_t, dim > a; 
        size_t i = 0; 
        for_each_combination(labels, labels + dim, labels + dim + 1, [&](auto b, [[maybe_unused]] auto e){
          a[i++] = rank_comb< colex, false >(b,_n,dim);
          return false; 
        });
        return std::tuple_cat(a);
      }
    }
  };

  [[nodiscard]]
  constexpr auto begin() const noexcept {
    return SimplexLabelIterator(s_labels.begin(), n);
  }
  [[nodiscard]]
  constexpr auto end() const noexcept {
    return SimplexLabelIterator(s_labels.end(), n);
  }
};

// template< uint8_t dim, typename index_t = uint16_t > 
// struct SimplexBoundaryRange {
//   static_assert(std::is_integral< index_t >::value, "Must be integral"); 
//   const vector< uint16_t > s_labels;
//   using iterator = typename vector< index_t >::iterator;
//   using const_iterator = typename vector< index_t >::const_iterator;

//   SimplexBoundaryRange(const vector< uint16_t > S) : s_labels(S) {}

//   struct SimplexBoundaryIterator {
//     const_iterator _it; 
//     SimplexBoundaryIterator(const_iterator it) : _it(it) {};
//     constexpr uint16_t* operator*() const noexcept { 
//       return (uint16_t*) &(*_it); 
//     }
//     constexpr void operator++() noexcept { _it += (dim+1); }
//     constexpr bool operator!=(SimplexBoundaryIterator o) const noexcept { return _it != o._it; }
//   };

//   [[nodiscard]]
//   constexpr auto begin() const noexcept {
//     return SimplexBoundaryIterator(s_labels.begin());
//   }
//   [[nodiscard]]
//   constexpr auto end() const noexcept {
//     return SimplexBoundaryIterator(s_labels.end());
//   }
// };

// struct binomial_coeff_table {
//   std::vector<std::vector<index_t>> B;
//   binomial_coeff_table(index_t n, index_t k) : B(k + 1, std::vector<index_t>(n + 1, 0)) {
//     for (index_t i = 0; i <= n; ++i) {
//       B[0][i] = 1;
//       for (index_t j = 1; j < std::min(i, k + 1); ++j)
//         B[j][i] = B[j - 1][i - 1] + B[j][i - 1];
//       if (i <= k) B[i][i] = 1;
//       //check_overflow(B[std::min(i >> 1, k)][i]);
//     }
//   }

//   index_t operator()(index_t n, index_t k) const {
//     assert(k < B.size() && n < B[k].size()  && n >= k - 1);
//     return B[k][n];
//   }
// };

// template <class Predicate>
// index_t get_max(index_t top, const index_t bottom, const Predicate pred) {
//   if (!pred(top)) {
//     index_t count = top - bottom;
//     while (count > 0) {
//       index_t step = count >> 1, mid = top - step;
//       if (!pred(mid)) {
//         top = mid - 1;
//         count -= step + 1;
//       } else {
//         count = step;
//       }
//     }
//   }
//   return top;
// }

// index_t get_max_vertex(const index_t idx, const index_t k, const index_t n, const binomial_coeff_table& B) {
//   return get_max(n, k - 1, [&](index_t w) -> bool { return (B(w, k) <= idx); });
// }

// template <typename OutputIterator>
// OutputIterator get_simplex_vertices(
//     index_t idx, 
//     const index_t dim, 
//     index_t n, 
//     OutputIterator out, 
//     const binomial_coeff_table& B
// ) {
//   --n;
//   for (index_t k = dim + 1; k > 1; --k) {
//     n = get_max_vertex(idx, k, n, B);
//     *out++ = n;
//     idx -= B(n, k);
//   }
//   *out = idx;
//   return out;
// }

// struct simplex_boundary_enumerator {
//     index_t idx_below, idx_above, j, k;
//     index_t dim;
//     const binomial_coeff_table& B;

//   public:
//     simplex_boundary_enumerator(const index_t i, const index_t _dim, const index_t n, const binomial_coeff_table& _bt)
//         : idx_below(i), idx_above(0), j(n - 1), k(_dim), B(_bt){}

//     simplex_boundary_enumerator(const index_t _dim, const index_t n, const binomial_coeff_table& _bt)
//         : simplex_boundary_enumerator(-1, _dim, n, _bt) {}

//     void set_simplex(const index_t i, const index_t _dim, const index_t n) {
//       idx_below = i;
//       idx_above = 0;
//       j = n - 1;
//       k = _dim;
//       dim = _dim;
//     }

//     bool has_next() { return (k >= 0); }

//     index_t next() {
//       j = get_max_vertex(idx_below, k + 1, j, B);
//       index_t face_index = idx_above - B(j, k + 1) + idx_below;
//       idx_below -= B(j, k + 1);
//       idx_above += B(j, k);
//       --k;
//       return face_index;
//     }
//   };