#pragma once
#include "attn/device.hpp"
#include "attn/registry.hpp"

// Registry entries, one per algorithm source file.
namespace attn::algs {

extern const AlgorithmSpec attn2d;  // attn3d.cpp
extern const AlgorithmSpec attn3d;  // attn3d.cpp
extern const AlgorithmSpec split;   // split.cpp
extern const AlgorithmSpec ring;    // ring.cpp

}  // namespace attn::algs
