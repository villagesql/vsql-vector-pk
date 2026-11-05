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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DECODED_VECTOR_CACHE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DECODED_VECTOR_CACHE_H

#include "../../native_vector.h"
#include "../../quantize.h"
#include "distance_evaluator.h"
#include "hnsw.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace svector::hnsw {

// Decoded vectors of graph nodes, keyed by vid, kept for the lifetime of one
// IndexGraph, i.e. one top-level call such as a single row's insert.
//
// An insert compares the same few nodes many times over: the Algorithm-4
// heuristic compares each candidate against every neighbour selected so far,
// and every neighbour whose list overflows reselects its own neighbours from
// its current list plus the new node. Without this cache each of those
// comparisons fetches the vector through the server and decodes it again. A
// vid's vector does not change while the call runs, so one decode serves
// every comparison in it.
//
// On an index with a VectorCache attached, a node that cache holds is kept
// here as a pointer to the cache's entry, which is never rewritten while the
// cache lives, so nothing is copied. Only a node the cache refused, or every
// node on an index without one, gets a decoded copy.
//
// Bytes live in fixed-size chunks that are never moved or freed until the
// cache is destroyed, and entries in an unordered_map whose elements do not
// move, so pointers from find() and add() stay valid for the cache's lifetime.
class DecodedVectorCache {
public:
  // A node's vector in the forms resolved so far. f32 is set for a node the
  // vector cache did not serve; q is the vector cache's entry for one it did,
  // or a quantized copy of f32 made for a comparison against such a node.
  struct Entry {
    const native::Data *f32 = nullptr;
    const quant::QData *q = nullptr;
  };

  // entry_size is the most bytes one decoded vector or quantized copy can
  // take; it is rounded up so every entry keeps native::Data's alignment.
  explicit DecodedVectorCache(size_t entry_size)
      : m_entry_size(round_up(entry_size)) {}

  // vid's entry, or nullptr if it has none yet.
  Entry *find(VID vid) {
    auto it = m_entries.find(vid.value);
    return it == m_entries.end() ? nullptr : &it->second;
  }

  // Records e as vid's entry.
  Entry &add(VID vid, Entry e) { return m_entries.emplace(vid.value, e).first->second; }

  // Records vid's entry from its raw stored value, decoded into the cache,
  // plus q, its quantized form if the caller already has one (the vector being
  // inserted, quantized once for its search). A no-op if vid already has an
  // entry. out, if given, is pointed at vid's entry either way. Returns true
  // on error.
  bool add_stored(VID vid, const Column::Data &stored, const quant::QData *q,
                  DistanceEvaluator &dist, std::span<char> err,
                  Entry **out = nullptr) {
    Entry *e = find(vid);
    if (e == nullptr) {
      const native::Data *decoded = nullptr;
      if (dist.decode(stored, reserve(), entry_size(), &decoded, err))
        return true;
      commit();
      e = &add(vid, {decoded, q});
    }
    if (out != nullptr)
      *out = e;
    return false;
  }

  // Distance between two entries. Two entries with an f32 form compare in
  // f32, even if a mixed pair earlier gave one a quantized copy, so a pair's
  // kernel never depends on what was compared before it. Otherwise at least
  // one is held by the vector cache, which has no f32 form, so the other is
  // brought to the quantized form too -- once, then kept in its entry.
  // Returns true on error.
  bool distance(Entry &a, Entry &b, DistanceEvaluator &dist, double &out,
                std::span<char> err) {
    if (a.f32 != nullptr && b.f32 != nullptr)
      return dist.eval(a.f32, b.f32, out, err);
    if ((a.q == nullptr && quantize(a, dist, err)) ||
        (b.q == nullptr && quantize(b, dist, err)))
      return true;
    return dist.eval(a.q, b.q, out, err);
  }

  // Bytes for the next decoded vector or quantized copy, entry_size() long.
  // They are not claimed until commit(), so a failed decode into them leaves
  // the cache unchanged and the same bytes are handed out again.
  void *reserve() {
    if (m_used == m_chunks.size() * ENTRIES_PER_CHUNK) {
      m_chunks.emplace_back(new std::byte[m_entry_size * ENTRIES_PER_CHUNK]);
    }
    return m_chunks[m_used / ENTRIES_PER_CHUNK].get() +
           (m_used % ENTRIES_PER_CHUNK) * m_entry_size;
  }

  // Claims the bytes the last reserve() returned.
  void commit() { ++m_used; }

  size_t entry_size() const { return m_entry_size; }

private:
  // Gives e, which must have an f32 form, a quantized copy in the cache.
  bool quantize(Entry &e, DistanceEvaluator &dist, std::span<char> err) {
    if (dist.quantize(e.f32->data, e.f32->dim, reserve(), entry_size(), &e.q,
                      err))
      return true;
    commit();
    return false;
  }

  static constexpr size_t ENTRIES_PER_CHUNK = 64;

  static size_t round_up(size_t n) {
    constexpr size_t align = alignof(std::max_align_t);
    return (n + align - 1) / align * align;
  }

  size_t m_entry_size;
  size_t m_used = 0;
  // new std::byte[] is aligned to max_align_t, which native::length() asks
  // for, and m_entry_size keeps every entry after the first aligned too.
  std::vector<std::unique_ptr<std::byte[]>> m_chunks;
  std::unordered_map<uint64_t, Entry> m_entries;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DECODED_VECTOR_CACHE_H
