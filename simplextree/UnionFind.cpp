//----------------------------------------------------------------------
//                        Disjoint-set data structure 
// File:                        union_find.cpp
//----------------------------------------------------------------------
// Copyright (c) 2018 Matt Piekenbrock. All Rights Reserved.
//
// Class definition based off of data-structure described here:  
// https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "UnionFind.h"
                   
void printCC(UnionFind& uf){
  for (size_t j = 0; j < uf.size; ++j){ py::print(uf.Find(j), " "); }
  py::print("\n");
}

PYBIND11_MODULE(_union_find, m) {
  py::class_<UnionFind>(m, "UnionFind")
    .def(py::init< size_t >())
    .def_readonly( "size", &UnionFind::size)
    .def_readonly( "parent", &UnionFind::parent)
    .def_readonly( "rank", &UnionFind::rank)
    .def("print", &printCC )
    .def("connected_components", &UnionFind::ConnectedComponents)
    .def("find", &UnionFind::Find)
    .def("find_all", &UnionFind::FindAll)
    .def("union", &UnionFind::Union)
    .def("union_all", &UnionFind::UnionAll)
    .def("add_sets", &UnionFind::AddSets)
    ;
}
