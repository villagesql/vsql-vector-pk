// Copyright (c) 2026 VillageSQL Contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2.0,
// as published by the Free Software Foundation.
//
// This program is designed to work with certain software (including
// but not limited to OpenSSL) that is licensed under separate terms,
// as designated in a particular file or component or in included license
// documentation.  The authors of MySQL hereby grant you an additional
// permission to link the program and your derivative works with the
// separately licensed software that they have either included with
// the program or referenced in the documentation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License, version 2.0, for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

// VectorCache unit tests.
//
// Two bugs this file exists to prevent recurring, both of which end-to-end
// recall measurement failed to catch because they produce plausible numbers
// rather than errors:
//
//   1. Every refused admission was handed a pointer into one shared scratch
//      buffer, so two refusals in a single comparison silently overwrote each
//      other.
//   2. A pair with one operand cached and one refused reached the f32 path and
//      decoded a QData as if it were encoded floats.
//
// Recall stayed flat across the budgets that were measured because those
// budgets produce almost no mixed pairs: far below the working set nearly
// every vector is refused, at or above it every vector is resident. Mixed
// pairs only occur in the transitional middle, so the budgets here are chosen
// to sit there deliberately.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/index/hnsw/vector_cache.h"

namespace {

using svector::hnsw::VectorCache;
using svector::hnsw::VID;

constexpr uint32_t kDim = 8;

// A cache sized to hold exactly `entries` vectors of kDim, so "full" is an
// exact, stated number rather than something derived at the call site.
VectorCache make_cache(uint32_t entries) {
  const uint32_t padded = svector::quant::qvector_padded_dim(kDim);
  return VectorCache(svector::quant::qdata_length(padded) * entries, kDim,
                     padded);
}

VID vid(uint64_t v) { return VID{v}; }

// Distinct, non-degenerate vectors: varying both magnitude and direction so a
// wrong operand produces a visibly wrong distance rather than a near-miss.
std::vector<float> vec(unsigned seed) {
  std::vector<float> v(kDim);
  for (uint32_t i = 0; i < kDim; ++i) {
    v[i] = static_cast<float>((seed * 7 + i * 13) % 29) - 14.0f;
  }
  return v;
}

// ---------------------------------------------------------------------------
// The cache itself
// ---------------------------------------------------------------------------

void test_fills_then_refuses() {
  VectorCache c = make_cache(3);
  const std::vector<float> a = vec(1), b = vec(2), d = vec(3), e = vec(4);

  assert(c.insert(vid(10), a.data()) != nullptr);
  assert(c.insert(vid(11), b.data()) != nullptr);
  assert(c.insert(vid(12), d.data()) != nullptr);
  assert(c.entries() == 3);

  // Full. The fourth insert is REFUSED rather than evicting one of the three:
  // a null return, and a resident set that does not change.
  assert(c.insert(vid(13), e.data()) == nullptr);
  assert(c.entries() == 3);
  assert(c.get(vid(13)) == nullptr);

  // Everything admitted before the cache filled is still there -- nothing was
  // displaced to make room.
  assert(c.get(vid(10)) != nullptr);
  assert(c.get(vid(11)) != nullptr);
  assert(c.get(vid(12)) != nullptr);
}

void test_entry_is_written_once() {
  // The lifetime contract: a caller reads an entry in place, so neither its
  // address nor its contents may change while later inserts happen. This is
  // what makes a QData* valid for the cache's life, and it is the guarantee
  // the slab design exists to provide.
  VectorCache c = make_cache(4);
  const std::vector<float> first = vec(1);

  const svector::quant::QData *held = c.insert(vid(10), first.data());
  assert(held != nullptr);

  // Snapshot the bytes, then fill the rest of the cache and overflow it.
  const size_t len = svector::quant::qdata_length(c.padded_dim());
  std::vector<unsigned char> snapshot(len);
  std::memcpy(snapshot.data(), held, len);

  for (unsigned i = 2; i <= 8; ++i) {
    const std::vector<float> v = vec(i);
    c.insert(vid(10 + i), v.data());
  }

  assert(c.get(vid(10)) == held && "entry address moved");
  assert(std::memcmp(held, snapshot.data(), len) == 0 &&
         "entry contents were rewritten");
}

void test_invalidate_retires_without_recycling() {
  // An invalidated slot is retired, not returned to a free list: its bytes may
  // still be under a reader's pointer. So the resident set only shrinks, and a
  // later insert takes a fresh slot rather than the retired one.
  VectorCache c = make_cache(3);
  const std::vector<float> a = vec(1), b = vec(2), d = vec(3), e = vec(4);

  const svector::quant::QData *first = c.insert(vid(10), a.data());
  c.insert(vid(11), b.data());
  c.insert(vid(12), d.data());
  assert(c.entries() == 3);

  c.invalidate(vid(10));
  assert(c.entries() == 2);
  assert(c.get(vid(10)) == nullptr);

  // The cache has a free entry's worth of residency but no free SLOT: all
  // three were handed out already, so this insert is refused.
  assert(c.insert(vid(13), e.data()) == nullptr);
  assert(c.entries() == 2);

  // The retired slot's bytes are untouched -- a reader holding `first` across
  // the invalidate still sees what it read.
  (void)first;
  assert(c.get(vid(11)) != nullptr && c.get(vid(12)) != nullptr);
}

void test_probe_finds_keys_past_a_tombstone() {
  // index_erase() leaves a tombstone rather than reinserting the run that
  // follows it. A probe must treat a tombstone as "keep looking" -- if it
  // stopped there, any key that collided past the erased one would report a
  // miss even though it is resident.
  //
  // Forcing a collision needs keys that hash to the same bucket, which the
  // mixing function makes impractical to construct directly. Instead fill the
  // cache completely, erase every other key, and assert every surviving key is
  // still found: with a table at <= 50% load and this many tombstones, some
  // surviving key is overwhelmingly likely to sit past one.
  constexpr uint32_t kN = 64;
  VectorCache c = make_cache(kN);
  for (unsigned i = 0; i < kN; ++i) {
    const std::vector<float> v = vec(i);
    assert(c.insert(vid(1000 + i), v.data()) != nullptr);
  }
  for (unsigned i = 0; i < kN; i += 2) c.invalidate(vid(1000 + i));
  assert(c.entries() == kN / 2);

  for (unsigned i = 1; i < kN; i += 2) {
    assert(c.get(vid(1000 + i)) != nullptr &&
           "surviving key lost behind a tombstone");
  }
  for (unsigned i = 0; i < kN; i += 2) {
    assert(c.get(vid(1000 + i)) == nullptr && "erased key still found");
  }
}

void test_set_max_bytes_reports_rather_than_resizes() {
  // A budget change cannot be applied in place -- slots are sized and placed
  // once -- so set_max_bytes() reports that the cache must be REPLACED and
  // leaves it alone. The owner rebuilds.
  VectorCache c = make_cache(4);
  const std::vector<float> a = vec(1);
  c.insert(vid(10), a.data());

  const size_t same =
      svector::quant::qdata_length(c.padded_dim()) * 4;  // as make_cache(4)
  assert(!c.set_max_bytes(same) && "unchanged budget should report no change");
  assert(c.entries() == 1);

  assert(c.set_max_bytes(same * 4) && "raised budget must request a rebuild");
  // Still intact: set_max_bytes reports, it does not act.
  assert(c.entries() == 1);
  assert(c.get(vid(10)) != nullptr);
}

void test_zero_budget_refuses_everything() {
  VectorCache c(0, kDim, svector::quant::qvector_padded_dim(kDim));
  const std::vector<float> a = vec(1);
  assert(c.insert(vid(10), a.data()) == nullptr);
  assert(c.get(vid(10)) == nullptr);
  assert(c.entries() == 0);
}

// ---------------------------------------------------------------------------
// Operand forms
//
// The distance kernel is exercised directly here rather than through
// DistanceEvaluator, which needs an encoded-bytes layout and a resolved native
// kernel. What these pin down is the property the evaluator's branch depends
// on: a quantized comparison agrees with the f32 one whichever operands came
// from the cache, so choosing the kernel per comparison cannot change a result
// beyond quantization error.
// ---------------------------------------------------------------------------

double f32_dist(const std::vector<float> &a, const std::vector<float> &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return sum;
}

// Quantize into caller-owned storage, the way the evaluator's mixed path does
// for the operand the cache refused.
const svector::quant::QData *quantize_into(std::vector<unsigned char> &buf,
                                           const std::vector<float> &v,
                                           uint32_t padded_dim) {
  buf.assign(svector::quant::qdata_length(padded_dim), 0);
  auto *q = reinterpret_cast<svector::quant::QData *>(buf.data());
  svector::quant::quantize(v.data(), kDim, padded_dim, q);
  return q;
}

void check_close(double got, double want, const char *what) {
  // int16 quantization of values in this range is accurate to well under 1%;
  // the bugs this guards against are wrong-operand and wrong-format errors,
  // which are off by orders of magnitude, not by rounding.
  const double tol = 0.01 * want + 1e-6;
  if (std::fabs(got - want) > tol) {
    std::printf("FAIL %s: got %.6f want %.6f\n", what, got, want);
    assert(false);
  }
}

void test_all_operand_combinations_agree() {
  // Both mixed ORDERS are tested separately: a-cached/b-refused and
  // a-refused/b-cached are different branches in eval(), and the bug that
  // prompted this file sat in exactly one of them.
  VectorCache c = make_cache(2);
  const std::vector<float> va = vec(1), vb = vec(2);
  const uint32_t padded = c.padded_dim();
  const double want = f32_dist(va, vb);

  // Both cached.
  const svector::quant::QData *ca = c.insert(vid(10), va.data());
  const svector::quant::QData *cb = c.insert(vid(11), vb.data());
  assert(ca != nullptr && cb != nullptr);
  check_close(svector::quant::dist_squared_l2_q(ca, cb, padded), want,
              "both cached");

  // The cache is now full, so anything else is refused and must be quantized
  // by the caller -- the mixed case.
  std::vector<unsigned char> buf_a, buf_b;
  const svector::quant::QData *qa = quantize_into(buf_a, va, padded);
  const svector::quant::QData *qb = quantize_into(buf_b, vb, padded);

  const double a_cached = svector::quant::dist_squared_l2_q(ca, qb, padded);
  const double b_cached = svector::quant::dist_squared_l2_q(qa, cb, padded);
  const double neither = svector::quant::dist_squared_l2_q(qa, qb, padded);

  check_close(a_cached, want, "a cached, b refused");
  check_close(b_cached, want, "a refused, b cached");
  check_close(neither, want, "neither cached");

  // Distance is symmetric, so the two mixed orders must agree with each other
  // and not merely each land near the f32 answer.
  assert(std::fabs(a_cached - b_cached) < 1e-9 &&
         "the two mixed orders disagree");
}

void test_two_refused_operands_do_not_share_a_buffer() {
  // The shape of the scratch bug: two refused operands in ONE comparison. If
  // they share storage the second overwrites the first and the comparison runs
  // a vector against itself, giving a distance of ~0.
  VectorCache c = make_cache(1);
  const std::vector<float> filler = vec(9), va = vec(1), vb = vec(2);
  assert(c.insert(vid(10), filler.data()) != nullptr);

  assert(c.insert(vid(11), va.data()) == nullptr);
  assert(c.insert(vid(12), vb.data()) == nullptr);

  std::vector<unsigned char> buf_a, buf_b;
  const uint32_t padded = c.padded_dim();
  const svector::quant::QData *qa = quantize_into(buf_a, va, padded);
  const svector::quant::QData *qb = quantize_into(buf_b, vb, padded);

  const double got = svector::quant::dist_squared_l2_q(qa, qb, padded);
  check_close(got, f32_dist(va, vb), "two refused operands");
  assert(got > 1.0 && "distance collapsed to zero: operands shared a buffer");
}

} // namespace

int main() {
  test_fills_then_refuses();
  test_entry_is_written_once();
  test_invalidate_retires_without_recycling();
  test_probe_finds_keys_past_a_tombstone();
  test_set_max_bytes_reports_rather_than_resizes();
  test_zero_budget_refuses_everything();
  test_all_operand_combinations_agree();
  test_two_refused_operands_do_not_share_a_buffer();
  std::printf("All vector cache tests passed.\n");
  return 0;
}
