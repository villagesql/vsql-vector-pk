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

// DecodedVectorCache holds one insert's node vectors. The vector being
// inserted is seeded into it by create_node() with the quantized form its
// search already made, so the comparisons that follow find it there instead of
// fetching it back -- and, with a vector cache attached, compare it the way the
// search did: f32 against a node the cache refused, quantized against one the
// cache holds, reusing that quantized form rather than making another.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "../src/index/hnsw/decoded_vector_cache.h"

namespace {

using svector::hnsw::Column;
using svector::hnsw::DecodedVectorCache;
using svector::hnsw::DistanceEvaluator;
using svector::hnsw::VID;
using Entry = DecodedVectorCache::Entry;

constexpr uint32_t kPrefix = 8;
constexpr uint32_t kDim = 37;
constexpr size_t kBufSize = kPrefix + kDim * sizeof(float);
constexpr uint32_t kPaddedDim = svector::quant::qvector_padded_dim(kDim);

// A stored value: [kPrefix bytes][little-endian float32 x dim].
struct Stored {
  explicit Stored(const std::vector<float> &v)
      : bytes(kPrefix + v.size() * sizeof(float)) {
    std::memset(bytes.data(), 0xAB, kPrefix);
    for (size_t i = 0; i < v.size(); i++) {
      svector::native::float4store(bytes.data() + kPrefix + i * sizeof(float),
                                   v[i]);
    }
    data.data = bytes.data();
    data.length = static_cast<uint32_t>(bytes.size());
  }
  std::vector<unsigned char> bytes;
  Column::Data data{};
};

// Quantized form held outside the cache under test, standing in for either a
// vector cache entry or the insert's prepared operand.
struct Quantized {
  explicit Quantized(const std::vector<float> &v)
      : bytes(svector::quant::qdata_length(kPaddedDim) / sizeof(uint32_t) + 1) {
    svector::quant::quantize(v.data(), kDim, kPaddedDim, get());
  }
  svector::quant::QData *get() {
    return reinterpret_cast<svector::quant::QData *>(bytes.data());
  }
  // uint32_t storage keeps QData's alignment.
  std::vector<uint32_t> bytes;
};

std::vector<float> make_vector(float seed) {
  std::vector<float> v(kDim);
  for (uint32_t i = 0; i < kDim; i++) {
    v[i] = std::sin(seed + 0.37f * i) * (1.0f + 0.1f * i);
  }
  return v;
}

double f32_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double sum = 0;
  for (uint32_t i = 0; i < kDim; i++) {
    const double d = static_cast<double>(a[i]) - b[i];
    sum += d * d;
  }
  return sum;
}

void check_close(double got, double want, double rel, const char *what) {
  if (std::fabs(got - want) > rel * want + 1e-6) {
    std::printf("FAIL %s: got %.9f want %.9f\n", what, got, want);
    assert(false);
  }
}

// An evaluator set up as on an index with a vector cache attached.
DistanceEvaluator make_evaluator() {
  DistanceEvaluator dist(svector::native::kDistL2Squared, kBufSize, kPrefix);
  dist.set_quantized(kPaddedDim);
  return dist;
}

DecodedVectorCache make_cache() {
  return DecodedVectorCache(std::max(
      kBufSize, DistanceEvaluator::max_qdata_length(kBufSize, kPrefix)));
}

// Seeding records the vector's f32 decode and keeps the caller's quantized
// form as is, without copying it.
void test_seed_keeps_both_forms() {
  DistanceEvaluator dist = make_evaluator();
  DecodedVectorCache cache = make_cache();
  char err[256] = {};
  const std::vector<float> v = make_vector(1.0f);
  Stored stored(v);
  Quantized prepared(v);

  Entry *e = nullptr;
  assert(!cache.add_stored(VID{7}, stored.data, prepared.get(), dist, err, &e));
  assert(e != nullptr && e == cache.find(VID{7}));
  assert(e->q == prepared.get());
  assert(e->f32 != nullptr && e->f32->dim == kDim);
  for (uint32_t i = 0; i < kDim; i++) {
    assert(e->f32->data[i] == v[i]);
  }
}

// create_node() seeds once per level; only the first call's entry is kept.
void test_seed_is_idempotent() {
  DistanceEvaluator dist = make_evaluator();
  DecodedVectorCache cache = make_cache();
  char err[256] = {};
  Stored first(make_vector(1.0f));
  Stored other(make_vector(2.0f));
  Quantized prepared(make_vector(1.0f));

  Entry *e1 = nullptr;
  Entry *e2 = nullptr;
  assert(!cache.add_stored(VID{7}, first.data, prepared.get(), dist, err, &e1));
  const auto *f32 = e1->f32;
  assert(!cache.add_stored(VID{7}, other.data, nullptr, dist, err, &e2));
  assert(e2 == e1);
  assert(e1->f32 == f32 && e1->q == prepared.get());
  assert(e1->f32->data[0] == make_vector(1.0f)[0]);
}

// Against a node the vector cache holds, the seeded entry runs the quantized
// kernel on its prepared form: no second quantize, so its q is unchanged.
void test_seed_vs_cached_uses_prepared_form() {
  DistanceEvaluator dist = make_evaluator();
  DecodedVectorCache cache = make_cache();
  char err[256] = {};
  const std::vector<float> va = make_vector(1.0f), vb = make_vector(2.0f);
  Stored stored_a(va);
  Quantized prepared_a(va);
  Quantized cached_b(vb);

  Entry *a = nullptr;
  assert(!cache.add_stored(VID{7}, stored_a.data, prepared_a.get(), dist, err,
                           &a));
  Entry &b = cache.add(VID{8}, {nullptr, cached_b.get()});

  double out = 0;
  assert(!cache.distance(*a, b, dist, out, err));
  assert(a->q == prepared_a.get());
  assert(out == svector::quant::dist_squared_l2_q(prepared_a.get(),
                                                  cached_b.get(), kPaddedDim));
  check_close(out, f32_l2(va, vb), 0.01, "seed vs cached");

  // Same in the other order.
  double rev = 0;
  assert(!cache.distance(b, *a, dist, rev, err));
  assert(std::fabs(rev - out) < 1e-9);
}

// Against a node the vector cache refused, the seeded entry compares in f32,
// as the insert's search did against that node, and nothing is quantized.
void test_seed_vs_refused_is_f32() {
  DistanceEvaluator dist = make_evaluator();
  DecodedVectorCache cache = make_cache();
  char err[256] = {};
  const std::vector<float> va = make_vector(1.0f), vb = make_vector(2.0f);
  Stored stored_a(va), stored_b(vb);
  Quantized prepared_a(va);

  Entry *a = nullptr;
  Entry *b = nullptr;
  assert(!cache.add_stored(VID{7}, stored_a.data, prepared_a.get(), dist, err,
                           &a));
  assert(!cache.add_stored(VID{8}, stored_b.data, nullptr, dist, err, &b));

  double out = 0;
  assert(!cache.distance(*a, *b, dist, out, err));
  assert(b->q == nullptr);
  double want = 0;
  assert(!dist.eval(stored_a.data, stored_b.data, want, err));
  assert(out == want);
  check_close(out, f32_l2(va, vb), 1e-5, "seed vs refused");
}

// A refused node compared against a cached one is quantized once and keeps
// that copy; it still compares in f32 against another refused node.
void test_refused_quantized_once_and_still_f32_with_refused() {
  DistanceEvaluator dist = make_evaluator();
  DecodedVectorCache cache = make_cache();
  char err[256] = {};
  const std::vector<float> va = make_vector(1.0f), vb = make_vector(2.0f),
                           vc = make_vector(3.0f);
  Stored stored_a(va), stored_c(vc);
  Quantized cached_b(vb);

  Entry *a = nullptr;
  Entry *c = nullptr;
  assert(!cache.add_stored(VID{7}, stored_a.data, nullptr, dist, err, &a));
  assert(!cache.add_stored(VID{9}, stored_c.data, nullptr, dist, err, &c));
  Entry &b = cache.add(VID{8}, {nullptr, cached_b.get()});

  double ab = 0;
  assert(!cache.distance(*a, b, dist, ab, err));
  const auto *a_q = a->q;
  assert(a_q != nullptr);
  check_close(ab, f32_l2(va, vb), 0.01, "refused vs cached");

  double ab2 = 0;
  assert(!cache.distance(b, *a, dist, ab2, err));
  assert(a->q == a_q && std::fabs(ab2 - ab) < 1e-9);

  double ac = 0;
  assert(!cache.distance(*a, *c, dist, ac, err));
  assert(c->q == nullptr);
  check_close(ac, f32_l2(va, vc), 1e-5, "refused vs refused");
}

} // namespace

int main() {
  test_seed_keeps_both_forms();
  test_seed_is_idempotent();
  test_seed_vs_cached_uses_prepared_form();
  test_seed_vs_refused_is_f32();
  test_refused_quantized_once_and_still_f32_with_refused();
  std::printf("All decoded vector cache tests passed.\n");
  return 0;
}
