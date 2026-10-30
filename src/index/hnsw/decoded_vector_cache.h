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
// Entries live in fixed-size chunks that are never moved or freed until the
// cache is destroyed, so a pointer returned by find() or commit() stays valid
// for the cache's lifetime.
class DecodedVectorCache {
public:
  // entry_size is the most bytes one decoded vector can take; it is rounded
  // up so every entry keeps native::Data's alignment.
  explicit DecodedVectorCache(size_t entry_size)
      : m_entry_size(round_up(entry_size)) {}

  // The cached decode of vid, or nullptr if it has none yet.
  const native::Data *find(VID vid) const {
    auto it = m_entries.find(vid.value);
    return it == m_entries.end() ? nullptr : it->second;
  }

  // Bytes for the next entry, entry_size() long. Nothing is recorded until
  // commit(), so a failed decode into them leaves the cache unchanged and the
  // same bytes are handed out again.
  void *reserve() {
    if (m_used == m_chunks.size() * ENTRIES_PER_CHUNK) {
      m_chunks.emplace_back(new std::byte[m_entry_size * ENTRIES_PER_CHUNK]);
    }
    return m_chunks[m_used / ENTRIES_PER_CHUNK].get() +
           (m_used % ENTRIES_PER_CHUNK) * m_entry_size;
  }

  // Records the bytes the last reserve() returned, now holding vid's decoded
  // vector.
  const native::Data *commit(VID vid) {
    const auto *data = static_cast<const native::Data *>(reserve());
    m_entries.emplace(vid.value, data);
    ++m_used;
    return data;
  }

  size_t entry_size() const { return m_entry_size; }

private:
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
  std::unordered_map<uint64_t, const native::Data *> m_entries;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_DECODED_VECTOR_CACHE_H
