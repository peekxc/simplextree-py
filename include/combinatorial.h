// combinatorial.h 
// Contains routines for combinatorics-related tasks 
// The combinations and permutations generation code is copyright Howard Hinnant, taken from: https://github.com/HowardHinnant/combinations/blob/master/combinations.h
#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H 

#include <cstdint>		// uint_fast64_t
#include <array>
// #include <span> 		 	// span (C++20)
#include <cmath>	 	 	// round, sqrt
#include <numeric>   	// midpoint, accumulate
#include <vector> 	 	// vector  
#include <algorithm> 
#include <type_traits>
#include <vector>
#include <functional>
#include <iterator>
#include <cassert>

using std::begin;
using std::end; 
using std::vector; 
using std::size_t;

namespace combinatorial {
	using index_t = uint_fast64_t;

	template<typename T>
	using it_diff_t = typename std::iterator_traits<T>::difference_type;

	// Rotates two discontinuous ranges to put *first2 where *first1 is.
	//     If last1 == first2 this would be equivalent to rotate(first1, first2, last2),
	//     but instead the rotate "jumps" over the discontinuity [last1, first2) -
	//     which need not be a valid range.
	//     In order to make it faster, the length of [first1, last1) is passed in as d1,
	//     and d2 must be the length of [first2, last2).
	//  In a perfect world the d1 > d2 case would have used swap_ranges and
	//     reverse_iterator, but reverse_iterator is too inefficient.
	template <class It>
	void rotate_discontinuous(
		It first1, It last1, it_diff_t< It > d1,
		It first2, It last2, it_diff_t< It > d2)
	{
		using std::swap;
		if (d1 <= d2){ std::rotate(first2, std::swap_ranges(first1, last1, first2), last2); }
		else {
			It i1 = last1;
			while (first2 != last2)
				swap(*--i1, *--last2);
			std::rotate(first1, i1, last1);
		}
	}

	// Call f() for each combination of the elements [first1, last1) + [first2, last2)
	//    swapped/rotated into the range [first1, last1).  As long as f() returns
	//    false, continue for every combination and then return [first1, last1) and
	//    [first2, last2) to their original state.  If f() returns true, return
	//    immediately.
	//  Does the absolute mininum amount of swapping to accomplish its task.
	//  If f() always returns false it will be called (d1+d2)!/(d1!*d2!) times.
	template < typename It, typename Lambda >
	bool combine_discontinuous(
		It first1, It last1, it_diff_t< It > d1,  
		It first2, It last2, it_diff_t< It > d2,
		Lambda&& f, it_diff_t< It > d = 0)
	{
		using D = it_diff_t< It >;
		using std::swap;
		if (d1 == 0 || d2 == 0){ return f(); }
		if (d1 == 1) {
			for (It i2 = first2; i2 != last2; ++i2) {
				if (f()){ return true; }
				swap(*first1, *i2);
			}
		}
		else {
			It f1p = std::next(first1), i2 = first2;
			for (D d22 = d2; i2 != last2; ++i2, --d22){
				if (combine_discontinuous(f1p, last1, d1-1, i2, last2, d22, f, d+1))
					return true;
				swap(*first1, *i2);
			}
		}
		if (f()){ return true; }
		if (d != 0){ rotate_discontinuous(first1, last1, d1, std::next(first2), last2, d2-1); }
		else { rotate_discontinuous(first1, last1, d1, first2, last2, d2); }
		return false;
	}

	template < typename Lambda, typename It > 
	struct bound_range { 
		Lambda f_;
		It first_, last_;
		bound_range(Lambda& f, It first, It last) : f_(f), first_(first), last_(last) {}
		bool operator()(){ return f_(first_, last_); } 
		bool operator()(It, It) { return f_(first_, last_); }
	};

	template <class It, class Function>
	Function for_each_combination(It first, It mid, It last, Function&& f) {
		bound_range<Function&, It> wfunc(f, first, mid);
		combine_discontinuous(first, mid, std::distance(first, mid),
													mid, last, std::distance(mid, last),
													wfunc);
		return std::move(f);
	}

	template <std::size_t... Idx>
	constexpr auto make_index_dispatcher(std::index_sequence<Idx...>) {
		return [] (auto&& f) { (f(std::integral_constant<std::size_t,Idx>{}), ...); };
	};
	
	template <std::size_t N>
	constexpr auto make_index_dispatcher() {
		return make_index_dispatcher(std::make_index_sequence< N >{});
	};

	template <typename T, size_t I > 
	struct tuple_n{
			template< typename...Args> using type = typename tuple_n<T, I-1>::template type<T, Args...>;
	};

	// Modified from: https://stackoverflow.com/questions/38885406/produce-stdtuple-of-same-type-in-compile-time-given-its-length-by-a-template-a
	template <typename T> 
	struct tuple_n<T, 0 > {
			template<typename...Args> using type = std::tuple<Args...>;   
	};
	template < typename T, size_t I >  using tuple_of = typename tuple_n<T, I>::template type<>;
	
	// Constexpr binomial coefficient using recursive formulation
	template < size_t n, size_t k >
	constexpr auto bc_recursive() noexcept {
		if constexpr ( n == k || k == 0 ){ return(1); }
		else if constexpr (n == 0 || k > n){ return(0); }
		else {
		 return (n * bc_recursive< n - 1, k - 1>()) / k;
		}
	}
	
	// Baseline from: https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
	// Requires O(min{k,n-k}), uses pascals triangle approach (+ degenerate cases)
	constexpr inline size_t binom(size_t n, size_t k) noexcept {
		return
			(        k> n  )? 0 :          // out of range
			(k==0 || k==n  )? 1 :          // edge
			(k==1 || k==n-1)? n :          // first
			(     k+k < n  )?              // recursive:
			(binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
			(binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
	}

	// Non-cached version of the binomial coefficient using floating point algorithm
	// Requires O(k), uses very simple loop
	[[nodiscard]]
	inline size_t binomial_coeff_(const double n, const size_t k) noexcept {
		// std::cout << "here2";
	  double bc = n;
	  for (size_t i = 2; i <= k; ++i){ bc *= (n+1-i)/i; }
	  return(static_cast< size_t >(std::round(bc)));
	}

	// Table to cache low values of the binomial coefficient
	template< size_t max_n, size_t max_k, typename value_t = index_t >
	struct BinomialCoefficientTable {
	  size_t pre_n = 0;
		size_t pre_k = 0; 
		value_t combinations[max_k][max_n+1];
	  vector< vector< value_t > > BT;

		constexpr BinomialCoefficientTable() : combinations() {
			auto n_dispatcher = make_index_dispatcher< max_n+1 >();
			auto k_dispatcher = make_index_dispatcher< max_k >();
			n_dispatcher([&](auto i) {
				k_dispatcher([&](auto j){
					combinations[j][i] = bc_recursive< i, j >();
				});
			});
	  }

		// Evaluate general binomial coefficient, using cached table if possible 
		value_t operator()(const index_t n, const index_t k) const {
			// std::cout << "INFO: " << pre_n << ", " << n <<  ":" << pre_k << ", " << k << " : " << BT.size() << std::endl; 
			if (n < max_n && k < max_k){ return combinations[k][n]; } // compile-time computed table
			if (n <= pre_n && k <= pre_k){ return BT[k][n]; } // runtime computed extension table
			if (k == 0 || n == k){ return 1; }
			if (n < k){ return 0; }
			if (k == 2){ return static_cast< value_t >((n*(n-1))/2); }
			if (k == 1){ return n; }
			// return binom(n, k);
			// return binomial_coeff_(n,std::min(k,n-k));
			return static_cast< value_t >(binomial_coeff_(n,std::min(k,n-k)));
		}

		void precompute(index_t n, index_t k){
			// std::cout << "here" << std::endl;
			pre_n = n;
			pre_k = k;
			BT = vector< vector< index_t > >(k + 1, vector< index_t >(n + 1, 0));
			for (index_t i = 0; i <= n; ++i) {
				BT[0][i] = 1;
				for (index_t j = 1; j < std::min(i, k + 1); ++j){
					BT[j][i] = BT[j - 1][i - 1] + BT[j][i - 1];
				}
				if (i <= k) { BT[i][i] = 1; };
			}
		}

		// Fast but unsafe access to a precompute table
		[[nodiscard]]
		constexpr auto at(index_t n, index_t k) noexcept -> index_t {
			return BT[k][n];
		}

	}; // BinomialCoefficientTable

	// Build the cached table
	static auto BC = BinomialCoefficientTable< 64, 3 >();
	static bool keep_table_alive = false; 

	// Wrapper to choose between cached and non-cached version of the Binomial Coefficient
	template< bool safe = true >
	constexpr size_t BinomialCoefficient(const size_t n, const size_t k){
		if constexpr(safe){
			return BC(n,k);
		} else {
			return BC.at(n,k);
		}
	}
	
	#if __cplusplus >= 202002L
    // C++20 (and later) code
		// constexpr midpoint midpoint
		using std::midpoint; 
	#else
		template < class Integer > 
		constexpr Integer midpoint(Integer a, Integer b) noexcept {
			return (a+b)/2;
		}
	#endif

	// All inclusive range binary search 
	// Compare must return -1 for <(key, index), 0 for ==(key, index), and 1 for >(key, index)
	// Guaranteed to return an index in [0, n-1] representing the lower_bound
	template< typename T, typename Compare > [[nodiscard]]
	int binary_search(const T key, size_t n, Compare p) noexcept {
	  int low = 0, high = n - 1, best = 0; 
		while( low <= high ){
			int mid = int{ midpoint(low, high) };
			auto cmp = p(key, mid);
			if (cmp == 0){ 
				while(p(key, mid + 1) == 0){ ++mid; }
				return(mid);
			}
			if (cmp < 0){ high = mid - 1; } 
			else { 
				low = mid + 1; 
				best = mid; 
			}
		}
		return(best);
	}
	
	// ----- Combinatorial Number System functions -----
	template< std::integral I, typename Compare > 
	void sort_contiguous(vector< I >& S, const size_t modulus, Compare comp){
		for (size_t i = 0; i < S.size(); i += modulus){
			std::sort(S.begin()+i, S.begin()+i+modulus, comp);
		}
	}

	// Lexicographically rank 2-subsets
	[[nodiscard]]
	constexpr auto rank_lex_2(index_t i, index_t j, const index_t n) noexcept {
	  if (j < i){ std::swap(i,j); }
	  return index_t(n*i - i*(i+1)/2 + j - i - 1);
	}

	// #include <iostream> 
	// Lexicographically rank k-subsets
	template< bool safe = true, typename InputIter >
	[[nodiscard]]
	inline index_t rank_lex_k(InputIter s, const size_t n, const size_t k, const index_t N){
		index_t i = k; 
		// std::cout << std::endl; 
	  const index_t index = std::accumulate(s, s+k, 0, [n, &i](index_t val, index_t num){ 
			// std::cout << BinomialCoefficient((n-1) - num, i) << ", ";
		  return val + BinomialCoefficient< safe >((n-1) - num, i--); 
		});
		// std::cout << std::endl; 
	  const index_t combinadic = (N-1) - index; // Apply the dual index mapping
	  return combinadic;
	}

	// Rank a stream of integers (lexicographically)
	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void rank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch (k){
			case 2:{
				for (; s != e; s += k){
					*out++ = rank_lex_2(*s, *(s+1), n);
				}
				break;
			}
			default: {
				const index_t N = BinomialCoefficient< safe >(n, k); 
				for (; s != e; s += k){
					*out++ = rank_lex_k< safe >(s, n, k, N);
				}
				break;
			}
		}
	}

	// should be in reverse colexicographical 
	template< bool safe = true >
	[[nodiscard]]
	constexpr auto rank_colex_2(index_t i, index_t j) noexcept {
		assert(i > j); // should be in colex order! 
		// return BinomialCoefficient< safe >(j, 2) + i;
		return j*(j-1)/2 + i;
		// const index_t index = std::accumulate(s, s+k, 0, [&i](index_t val, index_t num){ 
		// 	return val + BinomialCoefficient< safe >(num, i--); 
		// });
		// return index; 
	}
	
	// Colexicographically rank k-subsets
	// assumes each k tuple of s is in colex order! 
	template< bool safe = true, typename InputIter >
	[[nodiscard]]
	constexpr auto rank_colex_k(InputIter s, const size_t k) noexcept {
		index_t i = k; 
		const index_t index = std::accumulate(s, s+k, 0, [&i](index_t val, index_t num){ 
			return val + BinomialCoefficient< safe >(num, i--); 
		});
		return index; 
	}

	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void rank_colex(InputIt s, const InputIt e, [[maybe_unused]] const size_t n, const size_t k, OutputIt out){
		switch (k){
			case 2:{
				for (; s != e; s += k){
					*out++ = rank_colex_2(*s, *(s+1));
				}
				break;
			}
			default: {
				for (; s != e; s += k){
					*out++ = rank_colex_k< safe >(s, k);
				}
				break;
			}
		}
	}

	// colex bijection from a lexicographical order
	// index_t i = 1; 
	// const index_t index = std::accumulate(s, s+k, 0, [&i](index_t val, index_t num){ 
	// 	return val + BinomialCoefficient< safe >(num, i++); 
	// });
	// return index; 

	template< bool colex = true, bool safe = true, typename InputIt > 
	inline auto rank_comb(InputIt s, const size_t n, const size_t k){
		if constexpr(colex){
			return rank_colex_k< safe >(s, k);
		} else {
			const index_t N = BinomialCoefficient< safe >(n, k); 
			return rank_lex_k< safe >(s, n, k, N);
		}
	}

	template< bool colex = true, bool safe = true, typename InputIt, typename OutputIt > 
	inline void rank_combs(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		if constexpr(colex){
			for (; s != e; s += k){
				*out++ = rank_colex_k< safe >(s, k);
			}
		} else {
			const index_t N = BinomialCoefficient< safe >(n, k); 
			for (; s != e; s += k){
				*out++ = rank_lex_k< safe >(s, n, k, N);
			}
		}
	}

	// Lexicographically unrank 2-subsets
	template< typename OutputIt  >
	inline auto unrank_lex_2(const index_t r, const index_t n, OutputIt out) noexcept  {
		auto i = static_cast< index_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
		auto j = static_cast< index_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
		*out++ = i; // equivalent to *out = i; ++i;
		*out++ = j; // equivalent to *out = j; ++j;
	}
	
	// Lexicographically unrank k-subsets [ O(log n) version ]
	template< bool safe = true, typename OutputIterator > 
	inline void unrank_lex_k(index_t r, const size_t n, const size_t k, OutputIterator out) noexcept {
		const size_t N = combinatorial::BinomialCoefficient< safe >(n, k);
		r = (N-1) - r; 
		// auto S = std::vector< size_t >(k);
		for (size_t ki = k; ki > 0; --ki){
			int offset = binary_search(r, n, [ki](const auto& key, int index) -> int {
				auto c = combinatorial::BinomialCoefficient< safe >(index, ki);
				return(key == c ? 0 : (key < c ? -1 : 1));
			});
			r -= combinatorial::BinomialCoefficient< safe >(offset, ki); 
			*out++ = (n-1) - offset;
		}
	}

	// Lexicographically unrank subsets wrapper
	template< bool safe = true, typename InputIt, typename OutputIt >
	inline void unrank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch(k){
			case 2: 
				for (; s != e; ++s){ unrank_lex_2(*s, n, out); }
				break;
			default:
				for (; s != e; ++s){ unrank_lex_k< safe >(*s, n, k, out); }
				break;
		}
	}

	template <class Predicate>
	[[nodiscard]]
	index_t get_max(index_t top, const index_t bottom, const Predicate pred) noexcept {
		if (!pred(top)) {
			index_t count = top - bottom;
			while (count > 0) {
				index_t step = count >> 1, mid = top - step;
				if (!pred(mid)) {
					top = mid - 1;
					count -= step + 1;
				} else {
					count = step;
				}
			}
		}
		return top;
	}

	// From: Kruchinin, Vladimir, et al. "Unranking Small Combinations of a Large Set in Co-Lexicographic Order." Algorithms 15.2 (2022): 36.
	// return std::ceil(m * exp(log(r)/m + log(2*pi*m)/2*m + 1/(12*m*m) - 1/(360*pow(m,4)) - 1) + (m-1)/2);
	[[nodiscard]]
	constexpr auto find_k(index_t r, index_t m) noexcept -> int {
		if (r == 0 || m == 0){ return m - 1; }
		else if (m == 1){ return r - 1; }
		else if (m == 2){ return std::ceil(std::sqrt(1 + 8*r)/2) - 1; }
		else if (m == 3){ return std::ceil(std::pow(6*r, 1/3.)) - 1; }
		else { 
			return m - 1; 
		}
	}

	template< bool safe = true > 
	[[nodiscard]]
	index_t get_max_vertex(const index_t r, const index_t k, const index_t n) noexcept {
		// Binary searches in the range [k-1, n] for the largest index _w_ satisfying r >= C(w,k)
		// return get_max(n, k-1, [&](index_t w) -> bool { return r >= BinomialCoefficient< safe >(w, k); });
		
		const int lb = find_k(r,k);
		// assert(BinomialCoefficient(lb, k) <= r);
		return BinomialCoefficient< safe >(lb+1, k) > r ? 
			lb : 
				( BinomialCoefficient< safe >(lb+2, k) > r ? 
					lb + 1 : 
						( BinomialCoefficient< safe >(lb+3, k) > r ? 
							lb + 2 :  
							get_max(n, lb+3, [&](index_t w) -> bool { return r >= BinomialCoefficient< safe >(w, k); })
						)
				);
	}

	template < bool safe = true, typename InputIt, typename OutputIt >
	void unrank_colex(InputIt s, const InputIt e, const index_t n, const index_t k, OutputIt out) {
		for (index_t N = n - 1; s != e; ++s, N = n - 1){
			index_t r = *s; 
			for (index_t d = k; d > 1; --d) {
				N = get_max_vertex< safe >(r, d, N);
				// std::cout << "r: " << r << ", d: " << d << ", N: " << N << std::endl;
				*out++ = N;
				r -= BinomialCoefficient< safe >(N, d);
			}
			*out++ = r;
		}
	}

	// Unrank subsets wrapper
	template< bool colex = true, bool safe = true, typename InputIt, typename OutputIt >
	inline void unrank_combs(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		if constexpr(colex){
			unrank_colex< safe >(s, e, n, k, out);	
		} else {
			unrank_lex< safe >(s, e, n, k, out);
		}
	}

} // namespace combinatorial




	


//   // Lexicographically unrank subsets wrapper
// 	template< size_t k, typename InputIt, typename Lambda >
// 	inline void lex_unrank_f(InputIt s, const InputIt e, const size_t n, Lambda f){
//     if constexpr (k == 2){
//       std::array< I, 2 > edge;
//       for (; s != e; ++s){
//         lex_unrank_2(*s, n, edge.begin());
//         f(edge);
//       } 
//     } else if (k == 3){
//       std::array< I, 3 > triangle;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, 3, triangle.begin());
// 				f(triangle);
//       }
// 		} else {
//       std::array< I, k > simplex;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, k, simplex.begin());
// 				f(simplex);
//       }
//     }
// 	}

	
// 	[[nodiscard]]
// 	inline auto lex_unrank_2_array(const index_t r, const index_t n) noexcept -> std::array< I, 2 > {
// 		auto i = static_cast< index_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
// 		auto j = static_cast< index_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
// 		return(std::array< I, 2 >{ i, j });
// 	}
	
// 	[[nodiscard]]
// 	inline auto lex_unrank(const size_t rank, const size_t n, const size_t k) -> std::vector< index_t > {
// 		if (k == 2){
// 			auto a = lex_unrank_2_array(rank, n);
// 			std::vector< index_t > out(a.begin(), a.end());
// 			return(out);
// 		} else {
// 			std::vector< index_t > out; 
// 			out.reserve(k);
// 			lex_unrank_k(rank, n, k, std::back_inserter(out));
// 			return(out);
// 		}
// 	}
	

// 	template< typename Lambda >
// 	void apply_boundary(const size_t r, const size_t n, const size_t k, Lambda f){
// 		// Given a p-simplex's rank representing a tuple of size p+1, enumerates the ranks of its (p-1)-faces, calling Lambda(*) on its rank
// 		using combinatorial::I; 
// 		switch(k){
// 			case 0: { return; }
// 			case 1: {
// 				f(r);
// 				return;
// 			}
// 			case 2: {
// 				auto p_vertices = std::array< I, 2 >();
// 				lex_unrank_2(static_cast< index_t >(r), static_cast< index_t >(n), begin(p_vertices));
// 				f(p_vertices[0]);
// 				f(p_vertices[1]);
// 				return;
// 			}
// 			case 3: {
// 				auto p_vertices = std::array< I, 3 >();
// 				lex_unrank_k(r, n, k, begin(p_vertices));
// 				f(lex_rank_2(p_vertices[0], p_vertices[1], n));
// 				f(lex_rank_2(p_vertices[0], p_vertices[2], n));
// 				f(lex_rank_2(p_vertices[1], p_vertices[2], n));
// 				return; 
// 			} 
// 			default: {
// 				auto p_vertices = std::vector< index_t >(0, k);
// 				lex_unrank_k(r, n, k, p_vertices.begin());
// 				const index_t N = BinomialCoefficient(n, k); 
// 				combinatorial::for_each_combination(begin(p_vertices), begin(p_vertices)+2, end(p_vertices), [&](auto a, auto b){
// 					f(lex_rank_k(a, n, k, N));
// 					return false; 
// 				});
// 				return; 
// 			}
// 		}
// 	} // apply boundary 

// 	template< typename OutputIt >
// 	void boundary(const size_t p_rank, const size_t n, const size_t k, OutputIt out){
// 		apply_boundary(p_rank, n, k, [&out](auto face_rank){
// 			*out = face_rank;
// 			out++;
// 		});
// 	}
// } // namespace combinatorial

	// // Lexicographically unrank k-subsets
	// template< typename OutputIterator >
	// inline void lex_unrank_k(index_t r, const size_t k, const size_t n, OutputIterator out){
	// 	auto subset = std::vector< size_t >(k); 
	// 	size_t x = 1; 
	// 	for (size_t i = 1; i <= k; ++i){
	// 		while(r >= BinomialCoefficient(n-x, k-i)){
	// 			r -= BinomialCoefficient(n-x, k-i);
	// 			x += 1;
	// 		}
	// 		*out++ = (x - 1);
	// 		x += 1;
	// 	}
	// }

#endif 