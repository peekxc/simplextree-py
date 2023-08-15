#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "simplextree.h"
using simplex_t = SimplexTree::simplex_t; 

// Generic function to handle various vector types
template < typename Lambda >
void vector_handler(SimplexTree& st, const py::array_t< idx_t >& simplices, Lambda&& f){
  py::buffer_info s_buffer = simplices.request();
  if (s_buffer.ndim == 1){
    // py::print(s_buffer.shape[0]);
    const size_t n = s_buffer.shape[0]; 
    idx_t* s = static_cast< idx_t* >(s_buffer.ptr); 
    for (size_t i = 0; i < n; ++i){
      f(s+i, s+i+1);
    }
    // st.insert_it< true >(s, s+s_buffer.shape[0], st.root.get(), 0);
  } else if (s_buffer.ndim == 2) {
    // const size_t d = static_cast< size_t >(s_buffer.shape[1]);
    if (s_buffer.strides[0] <= 0){ return; }
    const size_t d = static_cast< size_t >(s_buffer.shape[1]);
    const size_t n =  static_cast< size_t >(s_buffer.shape[0]);
    idx_t* s = static_cast< idx_t* >(s_buffer.ptr); 
    // py::print("Strides: ", s_buffer.strides[0], s_buffer.strides[1], ", ", "size: ", s_buffer.size, ", shape: (", s_buffer.shape[0], s_buffer.shape[1], ")");
    for (size_t i = 0; i < n; ++i){
      f(s+(d*i), s+d*(i+1));
      // st.insert_it< true >(s+(d*i), s+(d*i)+1, st.root.get(), 0);
    }
  }
}

// TODO: accept py::buffer?
void insert_(SimplexTree& st, const py::array_t< idx_t >& simplices){
  vector_handler(st, simplices, [&st](idx_t* b, idx_t* e){
    // py::print(py::cast(simplex_t(b,e)));
    st.insert_it< true >(b, e, st.root.get(), 0);
  });
}
void insert_list(SimplexTree& st, std::list< simplex_t > L){
  for (auto s: L){ st.insert(simplex_t(s)); }
}

void remove_(SimplexTree& st, const py::array_t< idx_t >& simplices){
  vector_handler(st, simplices, [&st](idx_t* b, idx_t* e){ 
    st.remove(st.find(simplex_t(b, e))); 
  });
}
void remove_list(SimplexTree& st, std::list< simplex_t > L){
  for (auto s: L){ 
    st.remove(st.find(simplex_t(s))); 
  }
}

// Vectorized find 
[[nodiscard]]
auto find_(SimplexTree& st, const py::array_t< idx_t >& simplices) noexcept -> py::array_t< bool > {
  std::vector< bool > v;
  vector_handler(st, simplices, [&st, &v](idx_t* b, idx_t* e){
    node_ptr np = st.find(simplex_t(b, e));
    v.push_back(np != st.root.get() && np != nullptr);
  });
  return(py::cast(v));
}

[[nodiscard]]
auto find_list(SimplexTree& st, std::list< simplex_t > L) -> py::array_t< bool > {
  std::vector< bool > v;
  for (auto s: L){ 
    v.push_back(st.find(simplex_t(s))); 
  }
  return py::cast(v);
}


bool collapse_(SimplexTree& st, const vector< idx_t >& tau, const vector< idx_t >& sigma){
  return st.collapse(st.find(tau), st.find(sigma));
}

auto get_k_simplices(SimplexTree& st, const size_t k) -> py::array_t< idx_t > {
  vector< idx_t > res;
  if (st.n_simplexes.size() <= k){ return py::cast(res); }
  const size_t ns = st.n_simplexes.at(k); 
  res.reserve(ns*(k+1));
  auto tr = st::k_simplices< true >(&st, st.root.get(), k);
  traverse(tr, [&res](node_ptr cn, idx_t depth, simplex_t sigma){
    res.insert(res.end(), sigma.begin(), sigma.end());
    return true; 
  });
  py::array_t< idx_t > out = py::cast(res);
  std::array< size_t, 2 > shape = { ns, k+1 };
  return(out.reshape(shape));
}

// Retrieve the vertices by their label
vector< idx_t > get_vertices(const SimplexTree& st) {
  if (st.n_simplexes.size() == 0){ return vector< idx_t >(); } //IntegerVector(); }
  vector< idx_t > v;
  v.reserve(st.n_simplexes.at(0));
  for (auto& cn: st.root->children){ 
    v.push_back(cn->label); 
  }
  return(v);
}
auto get_edges(SimplexTree& st) -> py::array_t< idx_t > { return get_k_simplices(st, 1); }
auto get_triangles(SimplexTree& st) -> py::array_t< idx_t > { return get_k_simplices(st, 2); }
auto get_quads(SimplexTree& st) -> py::array_t< idx_t > { return get_k_simplices(st, 3); }


// Exports the 1-skeleton as an adjacency matrix 
// auto as_adjacency_matrix(SimplexTree& stp, int p = 0) -> py::array_t< idx_t > {
//   const auto& vertices = st.root->children; 
//   const size_t n = vertices.size();
//   res  = ..

//   // Fill in the adjacency matrix
//   size_t i = 0; 
//   for (auto& vi: vertices){
//     for (auto& vj: vi->children){
//       auto it = std::lower_bound(begin(vertices), end(vertices), vj->label, [](const node_uptr& cn, const idx_t label){
//         return cn->label < label; 
//       });
//       const size_t j = std::distance(begin(vertices), it); 
//       res.at(i, j) = res.at(j, i) = 1;
//     }
//     ++i;
//   }
//   return(res);
// }


auto degree_(SimplexTree& st, vector< idx_t > ids) -> py::array_t< idx_t > {
  vector< idx_t > res(ids.size());
  std::transform(begin(ids), end(ids), begin(res), [&st](int id){
    return st.degree(static_cast< idx_t >(id));
  });
  return py::cast(res);
}

py::array_t< idx_t > degree_default(SimplexTree& st) {
  return(degree_(st, get_vertices(st)));
}

py::list adjacent_(const SimplexTree& st, vector< idx_t > ids = vector< idx_t >()){
  if (ids.size() == 0){ ids = get_vertices(st); }
  vector< vector< idx_t > > res(ids.size());
  std::transform(begin(ids), end(ids), begin(res), [&st](int id){
    return st.adjacent_vertices(static_cast< idx_t >(id));
  });
  return py::cast(res);
}


void print_tree(const SimplexTree& st){ 
  py::scoped_ostream_redirect stream(
    std::cout,                               // std::ostream&
    py::module_::import("sys").attr("stdout") // Python output
  );
  st.print_tree(std::cout); 
}
void print_cousins(const SimplexTree& st){ 
  py::scoped_ostream_redirect stream(
    std::cout,                               // std::ostream&
    py::module_::import("sys").attr("stdout") // Python output
  );
  st.print_cousins(std::cout); 
}

auto simplex_counts(const SimplexTree& st) -> py::array_t< size_t >  {
  auto zero_it = std::find(st.n_simplexes.begin(), st.n_simplexes.end(), 0); 
  auto ne = std::vector< size_t >(st.n_simplexes.begin(), zero_it);
  return(py::cast(ne));
}


// void make_flag_filtration(Filtration* st, const NumericVector& D){
//   if (st->n_simplexes.size() <= 1){ return; }
//   const size_t ne = st->n_simplexes.at(1);
//   const auto v = st->get_vertices();
//   const size_t N = BinomialCoefficient(v.size(), 2);
//   if (size_t(ne) == size_t(D.size())){
//     vector< double > weights(D.begin(), D.end());
//     st->flag_filtration(weights, false);
//   } else if (size_t(D.size()) == size_t(N)){ // full distance vector passed in
//     auto edge_iter = st::k_simplices< true >(st, st->root.get(), 1);
//     vector< double > weights;
//     weights.reserve(ne);
//     st::traverse(edge_iter, [&weights, &D, &v](node_ptr np, idx_t depth, simplex_t sigma){
//       auto v1 = sigma[0], v2 = sigma[1];
//       auto idx1 = std::distance(begin(v), std::lower_bound(begin(v), end(v), v1));
//       auto idx2 = std::distance(begin(v), std::lower_bound(begin(v), end(v), v2));
//       auto dist_idx = to_natural_2(idx1, idx2, v.size());
//       weights.push_back(D[dist_idx]);
//       return true; 
//     });
//     st->flag_filtration(weights, false);
//   } else {
//     throw std::invalid_argument("Flag filtrations require a vector of distances for each edge or a 'dist' object");
//   }
// }
// 
// void test_filtration(Filtration* st, const size_t i){
//   auto ind = st->simplex_idx(i);
//   for (auto idx: ind){ Rcout << idx << ", "; }
//   Rcout << std::endl;
//   auto sigma = st->expand_simplex(begin(ind), end(ind));
//   for (auto& label: sigma){ Rcout << label << ", "; }
//   Rcout << std::endl; 
// }



// PYBIND11_MODULE(filtration_module, m) {
//   py::class_< SimplexTree >("SimplexTree")
//     .def(py::init<>());
//     // .method( "as_XPtr", &as_XPtr)
//     .property("n_simplices", &simplex_counts, "Gets simplex counts")
//     // .field_readonly("n_simplices", &SimplexTree::n_simplexes)
//     .property("dimension", &SimplexTree::dimension)
//     .property("id_policy", &SimplexTree::get_id_policy, &SimplexTree::set_id_policy)
//     // .property("vertices", &get_vertices, "Returns the vertex labels as an integer vector.")
//     // .property("edges", &get_edges, "Returns the edges as an integer matrix.")
//     // .property("triangles", &get_triangles, "Returns the 2-simplices as an integer matrix.")
//     // .property("quads", &get_quads, "Returns the 3-simplices as an integer matrix.")
//     .property("connected_components", &SimplexTree::connected_components)
//     // .def( "print_tree", &print_tree )
//     // .def( "print_cousins", &print_cousins )
//     .def( "clear", &SimplexTree::clear)
//     .def( "generate_ids", &SimplexTree::generate_ids)
//     .def( "reindex", &SimplexTree::reindex)
//     // .def( "adjacent", &adjacent_R)
//     // .def( "degree", &degree_R)
//     // .def( "insert",  &insert_R)
//     // .def( "insert_lex", &insert_lex)
//     // .def( "remove",  &remove_R)
//     // .def( "find", &find_R)
//     .def( "expand", &SimplexTree::expansion )
//     .def( "collapse", &collapse_R)
//     // .def( "vertex_collapse", (bool (SimplexTree::*)(idx_t, idx_t, idx_t))(&SimplexTree::vertex_collapse))
//     .def( "contract", &SimplexTree::contract)
//     .def( "is_tree", &SimplexTree::is_tree)
//     // .def( "as_adjacency_matrix", &as_adjacency_matrix)
//     // .def( "as_adjacency_list", &as_adjacency_list)
//     // .def( "as_edge_list", &as_edge_list)
//     ;
//   // Rcpp::class_< Filtration >("Filtration")
//   //   .derives< SimplexTree >("SimplexTree")
//   //   .constructor()
//   //   .method("init_tree", &init_filtration)
//   //   .field("included", &Filtration::included)
//   //   .property("current_index", &Filtration::current_index)
//   //   .property("current_value", &Filtration::current_value)
//   //   .property("simplices", &get_simplices, "Returns the simplices in the filtration")
//   //   .property("weights", &Filtration::weights, "Returns the weights in the filtration")
//   //   .property("dimensions", &Filtration::dimensions, "Returns the dimensions of the simplices in the filtration")
//   //   .method("flag_filtration", &make_flag_filtration, "Constructs a flag filtration")
//   //   .method("threshold_value", &Filtration::threshold_value)
//   //   .method("threshold_index", &Filtration::threshold_index)
//   //   ;
// }

// --- Begin functional exports + helpers ---

// #include <Rcpp/Benchmark/Timer.h>
// // [[Rcpp::export]]
// NumericVector profile(SEXP st){
//   Rcpp::XPtr< SimplexTree > st_ptr(st);
//   Timer timer;
//   timer.step("start");
//   st_ptr->expansion(2);
//   timer.step("expansion");
  
//   NumericVector res(timer);
//   const size_t n = 1000;
//   for (size_t i=0; i < size_t(res.size()); ++i) { res[i] = res[i] / n; }
//   return res;
// }


// bool contains_arg(vector< std::string > v, std::string arg_name){
//   return(std::any_of(v.begin(), v.end(), [&arg_name](const std::string arg){ 
//       return arg == arg_name;
//   }));
// };

// // From: https://stackoverflow.com/questions/56465550/how-to-concatenate-lists-in-rcpp
// List cLists(List x, List y) {
//   int nsize = x.size(); 
//   int msize = y.size(); 
//   List out(nsize + msize);

//   CharacterVector xnames = x.names();
//   CharacterVector ynames = y.names();
//   CharacterVector outnames(nsize + msize);
//   out.attr("names") = outnames;
//   for(int i = 0; i < nsize; i++) {
//     out[i] = x[i];
//     outnames[i] = xnames[i];
//   }
//   for(int i = 0; i < msize; i++) {
//     out[nsize+i] = y[i];
//     outnames[nsize+i] = ynames[i];
//   }
//   return(out);
// }

// The types of traversal supported
enum TRAVERSAL_TYPE { 
  PREORDER = 0, LEVEL_ORDER = 1, FACES = 2, COFACES = 3, COFACE_ROOTS = 4, K_SKELETON = 5, 
  K_SIMPLICES = 6, MAXIMAL = 7, LINK = 8
}; 
// const size_t N_TRAVERSALS = 9;

// Exports a list with the parameters for a preorder traversal
// py::list parameterize_(SimplexTree& st, vector< idx_t > sigma, std::string type, Rcpp::Nullable<List> args){

//   if (type == "preorder" || type == "dfs") { param_res["traversal_type"] = int(PREORDER); }
//   else if (type == "level_order" || type == "bfs") { param_res["traversal_type"] = int(LEVEL_ORDER); }
//   else if (type == "cofaces" || type == "star") { param_res["traversal_type"] = int(COFACES); }
//   else if (type == "coface_roots") { param_res["traversal_type"] = int(COFACE_ROOTS); }
//   else if (type == "link"){ param_res["traversal_type"] = int(LINK); }
//   else if (type == "k_skeleton" || type == "skeleton"){ param_res["traversal_type"] = int(K_SKELETON); }
//   else if (type == "k_simplices" || type == "maximal-skeleton"){ param_res["traversal_type"] = int(K_SIMPLICES); }
//   else if (type == "maximal"){ param_res["traversal_type"] = int(MAXIMAL); }
//   else if(type == "faces"){ param_res["traversal_type"] = int(FACES); }
//   else { stop("Iteration 'type' is invalid. Please use one of: preorder, level_order, faces, cofaces, star, link, skeleton, or maximal-skeleton"); }
//   param_res.attr("class") = "st_traversal";
//   return(param_res);
// }

using param_pack = typename std::tuple< SimplexTree*, node_ptr, TRAVERSAL_TYPE >;

// Traverse some aspect of the simplex tree, given parameters 
// template < class Lambda > 
// void traverse_switch(SimplexTree& st, param_pack&& pp, List args, Lambda&& f){
//   auto args_str = as< vector< std::string > >(args.names());
//   SimplexTree* st = get< 0 >(pp);
//   node_ptr init = get< 1 >(pp);
//   TRAVERSAL_TYPE tt = get< 2 >(pp);
//   switch(tt){
//     case PREORDER: {
//       auto tr = st::preorder< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case LEVEL_ORDER: {
//       auto tr = st::level_order< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case FACES: {
//       auto tr = st::faces< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case COFACES: {
//       auto tr = st::cofaces< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case COFACE_ROOTS: {
//       auto tr = st::coface_roots< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case K_SKELETON: {
//       if (!contains_arg(args_str, "k")){ stop("Expecting dimension 'k' to be passed."); }
//       idx_t k = args["k"];
//       auto tr = st::k_skeleton< true >(st, init, k);
//       traverse(tr, f);
//       break; 
//     }
//     case K_SIMPLICES: {
//       if (!contains_arg(args_str, "k")){ stop("Expecting dimension 'k' to be passed."); }
//       idx_t k = args["k"];
//       auto tr = st::k_simplices< true >(st, init, k);
//       traverse(tr, f);
//       break; 
//     }
//     case MAXIMAL: {
//       auto tr = st::maximal< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//     case LINK: {
//       auto tr = st::link< true >(st, init);
//       traverse(tr, f);
//       break; 
//     }
//   }
// }

// // To validate the traversal parameters
// param_pack validate_params(List args){
//   // Extract parameters 
//   auto args_str = as< vector< std::string > >(args.names());
  
//   // Extract tree
//   if (!contains_arg(args_str, ".ptr")){ stop("Simplex tree pointer missing."); }
//   SEXP xptr = args[".ptr"]; 
//   if (TYPEOF(xptr) != EXTPTRSXP || R_ExternalPtrAddr(xptr) == NULL){
//     stop("Invalid pointer to simplex tree.");
//   }
//   XPtr< SimplexTree > st(xptr); // Unwrap XPtr
  
//   // Extract initial simplex
//   node_ptr init = nullptr; 
//   if (!contains_arg(args_str, "sigma")){ init = st->root.get(); }
//   else {
//     IntegerVector sigma = args["sigma"];
//     init = st->find(sigma); 
//     if (init == nullptr){ init = st->root.get(); }
//   }
//   if (init == nullptr){ stop("Invalid starting simplex"); }
  
//   // Extract traversal type 
//   size_t tt = (size_t) args["traversal_type"];
//   if (tt < 0 || tt >= N_TRAVERSALS){ stop("Unknown traversal type."); }
  
//   return(std::make_tuple(static_cast< SimplexTree* >(st), init, static_cast< TRAVERSAL_TYPE >(tt)));
// }

void traverse_(SimplexTree& stree, const size_t order, py::function f, simplex_t init = simplex_t(), const size_t k = 0){
  node_ptr base = init.size() == 0 ? stree.root.get() : stree.find(init);
  // py::print("Starting from root?", init.size() == 0 ? "Y" : "N", ", k=",k);
  const auto apply_f = [&f](node_ptr cn, idx_t depth, simplex_t s){
    f(py::cast(s));
    return true; 
  };
  switch(order){
    case PREORDER: {
      auto tr = st::preorder< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case LEVEL_ORDER: {
      auto tr = st::level_order< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case FACES: {
      auto tr = st::faces< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case COFACES: {
      auto tr = st::cofaces< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case COFACE_ROOTS: {
      auto tr = st::coface_roots< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case K_SKELETON: {
      auto tr = st::k_skeleton< true >(&stree, base, k);
      traverse(tr, apply_f);
      break; 
    }
    case K_SIMPLICES: {
      auto tr = st::k_simplices< true >(&stree, base, k);
      traverse(tr, apply_f);
      break; 
    }
    case MAXIMAL: {
      auto tr = st::maximal< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
    case LINK: {
      auto tr = st::link< true >(&stree, base);
      traverse(tr, apply_f);
      break; 
    }
  }
}

// // [[Rcpp::export]]
// List ltraverse_R(List args, Function f){
//   List res = List(); 
//   auto run_Rf = [&f, &res](node_ptr cn, idx_t d, simplex_t tau){
//     res.push_back(f(wrap(tau)));
//     return(true);
//   };
//   traverse_switch(validate_params(args), args, run_Rf);
//   return(res);
// }


#include <chrono>
#include <random>

// void expand_f_bernoulli(SimplexTree& stx, const size_t k, const double p){
//   SimplexTree& st = *(Rcpp::XPtr< SimplexTree >(stx));  
  
//   // Random number generator
//   std::random_device random_device;
//   std::mt19937 random_engine(random_device());
//   std::uniform_real_distribution< double > bernoulli(0.0, 1.0);
  
//   // Perform Bernoulli trials for given k, with success probability p
//   st.expansion_f(k, [&](node_ptr parent, idx_t depth, idx_t label){
//     double q = bernoulli(random_engine);
//     if (p == 1.0 | q < p){ // if successful trial
//       std::array< idx_t, 1 > int_label = { label };
//       st.insert_it(begin(int_label), end(int_label), parent, depth);
//     }
//   });
// }



// Expands st conditionally based on f
// void expansion_f(SimplexTree& st, const size_t k, py::function f){
//   const auto do_expand = [&](node_ptr np, auto depth, auto label){
//     // get simplex
//     return true; 
//   };
//   st.expansion_f(k, f);
// }


// pip install --no-deps --no-build-isolation --editable .
// clang -Wall -fPIC -c src/simplicial/simplextree_module.cpp -std=c++17 -I/Users/mpiekenbrock/pbsig/extern/pybind11/include -I/Users/mpiekenbrock/simplicial/src/simplicial/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_simplextree, m) {
  
  py::class_<SimplexTree>(m, "SimplexTree")
    .def(py::init<>())
    .def_property_readonly("n_simplices", &simplex_counts)
    .def_property_readonly("dimension", &SimplexTree::dimension)
    .def_property("id_policy", &SimplexTree::get_id_policy, &SimplexTree::set_id_policy)
    .def_property_readonly("vertices", &get_vertices)
    .def_property_readonly("edges", &get_edges)
    .def_property_readonly("triangles", &get_triangles)
    .def_property_readonly("quads", &get_quads)
    .def_property_readonly("connected_components", &SimplexTree::connected_components)
    .def( "print_tree", &print_tree )
    .def( "print_cousins", &print_cousins )
    .def( "clear", &SimplexTree::clear)
    .def( "_degree", &degree_)
    .def( "_degree_default", &degree_default)
    // .def( "degree", static_cast< py::array_t< idx_t >(SimplexTree::*)() >(&degree_default), "degree")
    .def("_insert", &insert_)
    .def("_insert_list", &insert_list)
    // .def( "insert_lex", &insert_lex)
    .def( "_remove",  &remove_)
    .def( "_remove_list",  &remove_list)
    .def( "_find", &find_)
    .def( "_find_list", &find_list)
    .def( "_adjacent", &adjacent_)
    .def( "_collapse", &collapse_)
    .def( "generate_ids", &SimplexTree::generate_ids)
    .def( "_reindex", &SimplexTree::reindex)
    .def( "_expand", &SimplexTree::expansion )
    .def( "_vertex_collapse", (bool (SimplexTree::*)(idx_t, idx_t, idx_t))(&SimplexTree::vertex_collapse))
    .def( "_contract", &SimplexTree::contract)
    .def( "is_tree", &SimplexTree::is_tree)
    .def( "_traverse", &traverse_)
    // .def( "as_adjacency_matrix", &as_adjacency_matrix)
    ;
}