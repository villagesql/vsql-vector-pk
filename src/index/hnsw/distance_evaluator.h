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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DISTANCE_EVALUATOR_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DISTANCE_EVALUATOR_H

#include "../../distance_registry.h"
#include "../../native_vector.h"
#include "hnsw.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>

namespace svector::hnsw {

// Computes the graph distance between two raw stored vector values using a
// native kernel, owning everything that requires: the resolved kernel pointer
// and the scratch buffers for operands that cannot be read in place. Pulling
// this out of IndexGraph keeps that class about graph structure and gives the
// "read-and-compute" hot path -- including its fixed-operand reuse -- one home.
//
// A stored value is [storage-ref prefix][little-endian float32 x dim]. On a
// little-endian host those bytes already are the floats, so the kernel reads
// them in place when they are aligned for float; otherwise (a big-endian host,
// or a misaligned source) they are decoded into a scratch buffer first.
//
// Reuse model: in a layer traversal one operand (the query/insert vector) is
// fixed across every candidate. eval() therefore resolves operand `a` only when
// its source pointer differs from the last call, and always resolves `b`. The
// key is the raw source pointer: identical pointer => identical bytes. So `a`
// must never come from a buffer whose address is reused with DIFFERENT
// contents across calls, or a stale pointer-match would reuse the wrong
// vector.
class DistanceEvaluator {
public:
  using NativeDistFn = native::DistFn;

  // helper_name is the distance helper the profile bound at fn_id 1 (the caller
  // reads it from the server and passes it in); the evaluator maps it to a
  // native kernel via distance_registry.h. An unknown name leaves the evaluator
  // unresolved -- eval() then reports it, naming the offending helper. buf_size
  // is the raw maximum value length; the two scratch buffers are sized to it
  // (>= the decoded float length). prefix is the storage-ref header stripped
  // off each value before its floats.
  DistanceEvaluator(const char *helper_name, size_t buf_size, uint32_t prefix)
      : m_fn(native::dist_for_name(helper_name)), m_prefix(prefix),
        m_scratch_a(buf_size), m_scratch_b(buf_size) {
    if (m_fn == nullptr && helper_name != nullptr) {
      // Keep the name for eval()'s error; the buffer is small and fixed.
      std::strncpy(m_helper_name, helper_name, sizeof(m_helper_name) - 1);
    }
  }

  bool valid() const { return m_fn != nullptr; }

  // Resolve a and b to their floats (a reused per the model above), compute
  // the distance, write it to out. Returns true on error (writes err). err is a
  // non-owning buffer. out is double -- IndexGraph::DistanceType -- taken
  // concretely here to avoid a circular include of graph.h for what is a fixed
  // type.
  bool eval(const Column::Data &a, const Column::Data &b, double &out,
            std::span<char> err) {
    if (m_fn == nullptr) {
      return unresolved(err);
    }
    // Operand a is fixed across a traversal's candidates, so reuse it when the
    // source pointer is unchanged; b always varies.
    if (a.data != m_a_src) {
      if (floats(a, m_scratch_a, m_a_floats, m_a_dim, err)) {
        m_a_src = nullptr;
        return true;
      }
      m_a_src = a.data;
    }
    const float *b_floats = nullptr;
    uint32_t b_dim = 0;
    if (floats(b, m_scratch_b, b_floats, b_dim, err))
      return true;
    if (b_dim != m_a_dim) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: dimension mismatch (%u vs %u)", m_a_dim, b_dim);
      return true;
    }
    out = m_fn(m_a_floats, b_floats, m_a_dim);
    return false;
  }

  // Computes the distance between two operands the caller has already
  // decoded, e.g. held in a DecodedVectorCache. Returns true on error.
  bool eval(const native::Data *a, const native::Data *b, double &out,
            std::span<char> err) {
    if (m_fn == nullptr) {
      return unresolved(err);
    }
    out = m_fn(a->data, b->data, a->dim);
    return false;
  }

  // Decodes the raw stored value v into buf, which must be buf_len bytes
  // aligned for native::Data, and points *dp at it. Returns true on error.
  bool decode(const Column::Data &v, void *buf, size_t buf_len,
              const native::Data **dp, std::span<char> err) {
    if (v.length < m_prefix) {
      snprintf(err.data(), err.size(), "HNSW: distance: value too short (len=%u)",
               v.length);
      return true;
    }
    // TODO(villagesql): this "strip the [8-byte storage ref] prefix, then
    // native::from_encoded the floats" decode is duplicated here, in
    // svector_distance_impl (vector.cc), and in several length computations in
    // vector.cc that open-code sizeof(vef_storage_ref_t) + n*sizeof(float). The
    // persisted SVECTOR format ([ref][floats]) has no single owner; consolidate
    // all these sites into one format helper (e.g. svector_format.h) instead of
    // adding yet another copy.
    if (native::from_encoded(v.data + m_prefix, v.length - m_prefix, buf,
                             buf_len)) {
      snprintf(err.data(), err.size(), "HNSW: distance: failed to decode vector");
      return true;
    }
    *dp = static_cast<const native::Data *>(buf);
    return false;
  }

private:
  bool unresolved(std::span<char> err) const {
    snprintf(err.data(), err.size(),
             "HNSW: distance: unresolved distance function (helper '%s')",
             m_helper_name);
    return true;
  }

  // Points out at the dim floats of the raw stored value v: in place when the
  // host is little-endian and they are aligned for float, else decoded into
  // buf. Returns true on error.
  bool floats(const Column::Data &v, ScratchBytes &buf, const float *&out,
              uint32_t &dim, std::span<char> err) {
    if (v.length < m_prefix || (v.length - m_prefix) % sizeof(float) != 0 ||
        v.length - m_prefix > buf.size()) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: bad vector length (len=%u)", v.length);
      return true;
    }
    dim = (v.length - m_prefix) / sizeof(float);
    if (dim == 0 || dim > native::MAX_VECTOR_DIMENSION) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: bad vector dimension (%u)", dim);
      return true;
    }
    const unsigned char *src = v.data + m_prefix;
    if constexpr (std::endian::native == std::endian::little) {
      if (native::is_aligned(src, alignof(float))) {
        out = reinterpret_cast<const float *>(src);
        return false;
      }
    }
    auto *dst = reinterpret_cast<float *>(buf.data());
    for (uint32_t i = 0; i < dim; i++) {
      dst[i] = native::float4get(src + i * sizeof(float));
    }
    out = dst;
    return false;
  }

  NativeDistFn m_fn;
  uint32_t m_prefix;
  // Scratch for each operand when it cannot be read in place. Two buffers so
  // both are live at once for the kernel call.
  ScratchBytes m_scratch_a;
  ScratchBytes m_scratch_b;
  // Source pointer last resolved for operand a, and its floats; nullptr means
  // nothing reusable is held. See the reuse model above.
  const unsigned char *m_a_src = nullptr;
  const float *m_a_floats = nullptr;
  uint32_t m_a_dim = 0;
  // The bound helper name, kept only when it did not resolve, for eval()'s
  // error message. Same 64-byte cap the server name lookup uses.
  char m_helper_name[64] = {};
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DISTANCE_EVALUATOR_H
