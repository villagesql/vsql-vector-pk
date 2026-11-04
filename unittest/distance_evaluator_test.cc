// Copyright (c) 2026 VillageSQL Contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2.0,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License, version 2.0, for more details.

// Standalone, assert()-based unit test for the SVECTOR column storage layer
// (svector::MultiColumnStore / ColumnStore), driven WITHOUT InnoDB or the
// server. The storage primitives (Page/Segment/MtrCtx) in
// <villagesql/preview/storage_api.h> call through a vtable
// (vsql::preview_storage::detail::g_abi) that the server normally populates
// with InnoDB-backed functions. Here we install an in-memory fake backend --
// a map of page-number -> fixed-size page buffer -- so the real storage code
// can run in a plain unit-test process.
//
// PURPOSE: reproduce BUG 7 -- the multi-segment level-store page-allocation
// bug. The HNSW IndexStore lays out per-level stores across 2 segments

// DistanceEvaluator reads a stored vector's floats in place when it can, and
// decodes them into scratch otherwise. Both paths must give the same distance
// as computing it from the floats directly, for every registered metric.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "../src/index/hnsw/distance_evaluator.h"

namespace {

using svector::hnsw::Column;
using svector::hnsw::DistanceEvaluator;

constexpr uint32_t kPrefix = 8;
constexpr uint32_t kDim = 37;
constexpr size_t kBufSize = kPrefix + kDim * sizeof(float);

// A stored value: [kPrefix bytes][little-endian float32 x dim], written at
// offset `shift` into its own storage so a test can make it misaligned.
struct Stored {
  Stored(const std::vector<float> &v, size_t shift)
      : bytes(shift + kPrefix + v.size() * sizeof(float) + 16) {
    unsigned char *p = bytes.data() + shift;
    std::memset(p, 0xAB, kPrefix);
    for (size_t i = 0; i < v.size(); i++) {
      svector::native::float4store(p + kPrefix + i * sizeof(float), v[i]);
    }
    data.data = p;
    data.length = static_cast<uint32_t>(kPrefix + v.size() * sizeof(float));
  }
  std::vector<unsigned char> bytes;
  Column::Data data{};
};

std::vector<float> make_vector(float seed) {
  std::vector<float> v(kDim);
  for (uint32_t i = 0; i < kDim; i++) {
    v[i] = std::sin(seed + 0.37f * i) * (1.0f + 0.1f * i);
  }
  return v;
}

double expected(const char *metric, const std::vector<float> &a,
                const std::vector<float> &b) {
  double l1 = 0, l2 = 0, dot = 0, na = 0, nb = 0;
  for (uint32_t i = 0; i < kDim; i++) {
    double x = a[i], y = b[i];
    l1 += std::fabs(x - y);
    l2 += (x - y) * (x - y);
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (std::strcmp(metric, svector::native::kDistL1) == 0)
    return l1;
  if (std::strcmp(metric, svector::native::kDistL2Squared) == 0)
    return l2;
  if (std::strcmp(metric, svector::native::kDistCosine) == 0)
    return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
  return dot;
}

bool close(double got, double want) {
  return std::fabs(got - want) <= 1e-4 * std::max(1.0, std::fabs(want));
}

constexpr const char *kMetrics[] = {
    svector::native::kDistL1, svector::native::kDistL2Squared,
    svector::native::kDistCosine, svector::native::kDistInnerProduct};

// Aligned operands are read in place; operands shifted by 1 byte cannot be,
// and take the decode path. All four combinations must agree.
void test_in_place_and_decoded_agree() {
  const auto va = make_vector(0.5f);
  const auto vb = make_vector(2.0f);
  for (const char *metric : kMetrics) {
    for (size_t shift_a : {0u, 1u}) {
      for (size_t shift_b : {0u, 1u}) {
        Stored a(va, shift_a);
        Stored b(vb, shift_b);
        DistanceEvaluator eval(metric, kBufSize, kPrefix);
        char err[128] = {};
        double out = 0;
        assert(!eval.eval(a.data, b.data, out, err));
        if (!close(out, expected(metric, va, vb))) {
          fprintf(stderr, "%s shift %zu/%zu: got %.9g want %.9g\n", metric,
                  shift_a, shift_b, out, expected(metric, va, vb));
          assert(false);
        }
      }
    }
  }
}

// Operand a is reused while its source pointer is unchanged, and re-read when
// it changes; b is read on every call.
void test_fixed_operand_reuse() {
  const auto va = make_vector(0.5f);
  const auto vb = make_vector(2.0f);
  const auto vc = make_vector(4.0f);
  for (size_t shift : {0u, 1u}) {
    Stored a(va, shift);
    Stored b(vb, shift);
    Stored c(vc, shift);
    DistanceEvaluator eval(svector::native::kDistL2Squared, kBufSize, kPrefix);
    char err[128] = {};
    double out = 0;
    const char *l2 = svector::native::kDistL2Squared;
    assert(!eval.eval(a.data, b.data, out, err));
    assert(close(out, expected(l2, va, vb)));
    assert(!eval.eval(a.data, c.data, out, err));
    assert(close(out, expected(l2, va, vc)));
    assert(!eval.eval(c.data, b.data, out, err));
    assert(close(out, expected(l2, vc, vb)));
  }
}

void test_errors() {
  const auto va = make_vector(0.5f);
  Stored a(va, 0);
  char err[128] = {};
  double out = 0;

  DistanceEvaluator unknown("no_such_distance", kBufSize, kPrefix);
  assert(!unknown.valid());
  assert(unknown.eval(a.data, a.data, out, err));
  assert(std::strstr(err, "no_such_distance") != nullptr);

  DistanceEvaluator eval(svector::native::kDistL2Squared, kBufSize, kPrefix);
  // Dimensions differ.
  Stored shorter(std::vector<float>(va.begin(), va.end() - 1), 0);
  assert(eval.eval(a.data, shorter.data, out, err));
  assert(std::strstr(err, "dimension mismatch") != nullptr);
  // Shorter than the prefix, and not a whole number of floats. They get
  // their own storage: a pointer already resolved as operand a must keep its
  // bytes (see the reuse model).
  Stored bad(va, 0);
  Column::Data tiny{bad.data.data, kPrefix - 1};
  assert(eval.eval(a.data, tiny, out, err));
  Column::Data ragged{bad.data.data, kPrefix + 5};
  assert(eval.eval(a.data, ragged, out, err));
  // A failed operand a is not reused by the next call.
  assert(eval.eval(ragged, a.data, out, err));
  assert(eval.eval(ragged, a.data, out, err));
  assert(!eval.eval(a.data, a.data, out, err));
  assert(out == 0);
}

} // namespace

int main() {
  test_in_place_and_decoded_agree();
  test_fixed_operand_reuse();
  test_errors();
  printf("distance_evaluator_test: OK\n");
  return 0;
}
