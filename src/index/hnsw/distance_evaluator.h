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
#include "../../quantize.h"
#include "hnsw.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>

namespace svector::hnsw {

// Computes the graph distance between two raw stored vector values using a
// native kernel, owning everything that requires: the resolved kernel pointer
// and the scratch buffers the raw [storage-ref prefix][floats] operands are
// decoded into. Pulling this out of IndexGraph keeps that class about graph
// structure and gives the "decode-and-compute" hot path -- including its
// fixed-operand decode reuse -- one home.
//
// Reuse model: in a layer traversal one operand (the query/insert vector) is
// fixed across every candidate. eval() therefore decodes operand `a` only when
// its source pointer differs from the last call, and always decodes `b`. The
// key is the raw source pointer: identical pointer => identical already-decoded
// bytes. Callers that feed `a` from a buffer whose address is reused with
// DIFFERENT contents across calls must call invalidate_fixed_operand() first,
// or a stale pointer-match would reuse the wrong decode.
class DistanceEvaluator {
public:
  using NativeDistFn = native::DistFn;

  // helper_name is the distance helper the profile bound at fn_id 1 (the caller
  // reads it from the server and passes it in); the evaluator maps it to a
  // native kernel via distance_registry.h. An unknown name leaves the evaluator
  // unresolved -- eval() then reports it, naming the offending helper. buf_size
  // is the raw maximum value length; the two decode buffers are sized to it (>=
  // the decoded native length). prefix is the storage-ref header stripped off
  // each value before decoding the floats.
  DistanceEvaluator(const char *helper_name, size_t buf_size, uint32_t prefix)
      : m_fn(native::dist_for_name(helper_name)), m_prefix(prefix),
        m_decoded_a(buf_size), m_decoded_b(buf_size), m_qbuf_a(buf_size),
        m_qbuf_b(buf_size) {
    if (m_fn == nullptr && helper_name != nullptr) {
      // Keep the name for eval()'s error; the buffer is small and fixed.
      std::strncpy(m_helper_name, helper_name, sizeof(m_helper_name) - 1);
    }
  }

  bool valid() const { return m_fn != nullptr; }

  // Switch to quantized operands. Once set, eval() treats both operands as
  // pointers to a quant::QData rather than encoded bytes: no decode, no copy,
  // the kernel reads them in place. Set once when an index attaches a cache,
  // never per call -- every operand on a cache-enabled index is quantized, hit
  // or miss, so results do not depend on cache state.
  //
  // L2 only for now: quantize.h has dist_squared_l2_q and nothing else, and
  // IndexStore::vector_cache() refuses to attach a cache on other metrics.
  void set_quantized(uint32_t padded_dim) {
    m_quantized = true;
    m_padded_dim = padded_dim;
  }

  // Record the quantized form of the fixed operand, produced once per search
  // by quantize_operand(). Held so a comparison against a CACHED vector can
  // use it while one against an uncached vector still uses the f32 decode --
  // the two forms of operand a coexist for the whole search.
  // `f32_src` is the encoded form of the same vector, so eval() can recognise
  // that an operand handed to it IS this prepared one rather than some node's
  // bytes that merely happen to be unquantized.
  void set_quantized_fixed_operand(const quant::QData *qa,
                                   const unsigned char *f32_src) {
    m_qa = qa;
    m_fixed_f32_src = f32_src;
  }
  bool quantized() const { return m_quantized; }

  // Decode an encoded value into the caller's buffer. Public so the cache fill
  // path can decode before quantizing, reusing the one place that knows the
  // persisted [ref][floats] layout.
  bool decode_encoded(const Column::Data &v, ScratchBytes &buf,
                      const native::Data **out, std::span<char> err) {
    return decode(v, buf, out, err);
  }

  // Quantize an encoded operand into the caller's buffer and return it as a
  // QData view. Used for the two vectors that never pass through the cache
  // because they are not in the index: the query vector, and the vector being
  // inserted. Both would otherwise be f32 against quantized neighbours.
  bool quantize_operand(const Column::Data &v, ScratchBytes &buf,
                        Column::Data &out, std::span<char> err) {
    const native::Data *d = nullptr;
    if (decode(v, m_decoded_b, &d, err)) return true;
    const size_t need = quant::qdata_length(m_padded_dim);
    if (buf.size() < need) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: quantize buffer too small (%zu < %zu)",
               buf.size(), need);
      return true;
    }
    auto *q = reinterpret_cast<quant::QData *>(buf.data());
    quant::quantize(d->data, d->dim, m_padded_dim, q);
    out.data = reinterpret_cast<const unsigned char *>(q);
    out.length = static_cast<uint32_t>(need);
    return false;
  }

  // Invalidate the fixed-operand (a) decode cache. Call before an eval() whose
  // `a` comes from a reused buffer that now holds different bytes at the same
  // address.
  void invalidate_fixed_operand() { m_decoded_a_src = nullptr; }

  // Decode a and b (a reused per the model above), compute the distance, write
  // it to out. Returns true on error (writes err). err is a non-owning buffer.
  // out is double -- IndexGraph::DistanceType -- taken concretely here to avoid
  // a circular include of graph.h for what is a fixed type.
  // `b_quantized` says which form operand b is in: true for a vector served
  // by the cache, false for one read from a page. It is a per-COMPARISON
  // choice, not a per-index one, so a vector the cache refused costs a page
  // read and nothing more -- quantizing it only to throw the result away
  // would make a full cache more expensive than no cache at all.
  bool eval(const Column::Data &a, const Column::Data &b, bool a_quantized,
            bool b_quantized, double &out, std::span<char> err) {
    if (m_fn == nullptr) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: unresolved distance function (helper '%s')",
               m_helper_name);
      return true;
    }
    // A quantized operand has no f32 form to fall back to -- the cache holds
    // only the QData -- so once EITHER operand is quantized the comparison has
    // to run the quantized kernel, and the other operand must be brought into
    // that form.
    //
    // Three ways that resolves, cheapest first:
    //   both quantized          -- the common case once the cache is warm
    //   a is the fixed operand  -- its quantized copy was made once per search
    //   mixed                   -- quantize the unquantized one here
    //
    // Only the third costs anything, and only on a pair where one vector is
    // resident and the other was refused. Quantizing EVERY uncached read
    // instead (the previous design) measured 3.6x slower on build and 0.41x on
    // queries once the cache was full, because the result was discarded every
    // time.
    //
    // TODO(villagesql-indexing): the mixed pair quantizes the uncached operand
    // and throws the result away. It is the only direction available today --
    // a cache entry keeps no f32 form, so the alternative is a page fetch to
    // undo work the cache already did -- but neither side is free. Mixed pairs
    // only occur while the cache is partially filled, so this is unmeasured at
    // scale; worth revisiting if a workload sits in that state (a large index
    // under a budget it never quite fills, or steady growth past a full one).
    if (a_quantized || b_quantized) {
      const quant::QData *qa = nullptr;
      const quant::QData *qb = nullptr;
      if (a_quantized) {
        qa = reinterpret_cast<const quant::QData *>(a.data);
      } else if (m_qa != nullptr && a.data == m_fixed_f32_src) {
        qa = m_qa;
      } else if (to_qdata(a, m_qbuf_a, &qa, err)) {
        return true;
      }
      if (b_quantized) {
        qb = reinterpret_cast<const quant::QData *>(b.data);
      } else if (to_qdata(b, m_qbuf_b, &qb, err)) {
        return true;
      }
      out = quant::dist_squared_l2_q(qa, qb, m_padded_dim);
      return false;
    }

    const native::Data *a_data = nullptr;
    const native::Data *b_data = nullptr;
    // Operand a is fixed across a traversal's candidates, so reuse its decode
    // when the source pointer is unchanged; b always varies.
    if (a.data == m_decoded_a_src) {
      a_data = reinterpret_cast<const native::Data *>(m_decoded_a.data());
    } else {
      if (decode(a, m_decoded_a, &a_data, err)) return true;
      m_decoded_a_src = a.data;
    }
    if (decode(b, m_decoded_b, &b_data, err)) return true;
    out = m_fn(a_data, b_data);
    return false;
  }

private:
  // Decode an encoded operand and quantize it into `buf`, for the mixed case
  // where the other operand is already quantized. `buf` is per-operand, so the
  // two sides of one comparison cannot overwrite each other.
  bool to_qdata(const Column::Data &v, ScratchBytes &buf,
                const quant::QData **out, std::span<char> err) {
    const native::Data *d = nullptr;
    if (decode(v, m_decoded_b, &d, err)) return true;
    const size_t need = quant::qdata_length(m_padded_dim);
    if (buf.size() < need) {
      snprintf(err.data(), err.size(),
               "HNSW: distance: quantize buffer too small (%zu < %zu)",
               buf.size(), need);
      return true;
    }
    auto *q = reinterpret_cast<quant::QData *>(buf.data());
    quant::quantize(d->data, d->dim, m_padded_dim, q);
    *out = q;
    return false;
  }

  bool decode(const Column::Data &v, ScratchBytes &buf,
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
    if (native::from_encoded(v.data + m_prefix, v.length - m_prefix, buf.data(),
                             buf.size())) {
      snprintf(err.data(), err.size(), "HNSW: distance: failed to decode vector");
      return true;
    }
    *dp = reinterpret_cast<const native::Data *>(buf.data());
    return false;
  }

  NativeDistFn m_fn;
  uint32_t m_prefix;
  // Decoded native::Data for each operand. Two buffers so both are live at once
  // for the kernel call.
  ScratchBytes m_decoded_a;
  ScratchBytes m_decoded_b;
  // Quantize scratch for the mixed case; one per operand so a comparison's two
  // sides never share a buffer. Sized like the decode buffers, from the
  // indexed column's maximum value length, which a QData always fits in.
  ScratchBytes m_qbuf_a;
  ScratchBytes m_qbuf_b;
  // Source pointer last decoded into m_decoded_a; nullptr means nothing
  // reusable is held. See the reuse model above.
  const unsigned char *m_decoded_a_src = nullptr;

  // Set by set_quantized(): operands are quant::QData, not encoded bytes.
  bool m_quantized = false;
  const quant::QData *m_qa = nullptr;
  const unsigned char *m_fixed_f32_src = nullptr;
  uint32_t m_padded_dim = 0;
  // The bound helper name, kept only when it did not resolve, for eval()'s
  // error message. Same 64-byte cap the server name lookup uses.
  char m_helper_name[64] = {};
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DISTANCE_EVALUATOR_H
