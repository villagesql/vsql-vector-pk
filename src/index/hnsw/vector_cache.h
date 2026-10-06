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

// VectorCache: a byte-bounded, VID-keyed cache of encoded vectors, sitting
// behind IndexGraph::resolve_node_data() -- the single funnel every vector read
// on the paged search path goes through.
//
// Why: profiling the paged path showed the per-neighbour vector fetch (an ABI
// call into the server, a column-store page lookup and a copy out) as a leading
// cost, co-equal with the distance math itself. A hit here replaces all of that
// with a pointer to bytes we already hold.
//
// What it does NOT do: the graph (node records, neighbour lists) stays on
// pages. This is the vector-only cache mode; the pointer-graph modes are a
// separate backend.
//
// Lifetime contract, which the caller depends on: a cached entry's bytes have a
// STABLE address for as long as it is resident, and entries are only ever
// evicted or invalidated between queries, never during one. That matters
// because DistanceEvaluator keys its decoded-operand reuse on the source
// pointer -- see resolve_node_data().
//
// NOT thread-safe. Single-threaded use only, matching the rest of the index
// today; concurrency lands with the wider productionization work.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "hnsw.h"
#include "stats.h"

namespace svector::hnsw {

class VectorCache {
public:
  // A resident entry's bytes, as handed back to the caller. Valid until the
  // entry is evicted or invalidated (see the lifetime contract above).
  struct Entry {
    const unsigned char *data = nullptr;
    uint32_t length = 0;
  };

  explicit VectorCache(size_t max_bytes) : m_max_bytes(max_bytes) {}

  // Look up vid. On a hit, fills out and returns true; the LRU position is
  // refreshed. On a miss, leaves out untouched and returns false -- the caller
  // fetches and then calls insert().
  bool get(VID vid, Entry &out);

  // Make vid resident with a copy of [data, data+length). Evicts least-recently
  // used entries first if that would exceed the budget. A vector larger than
  // the whole budget is simply not cached (and not an error: the caller already
  // holds the bytes it fetched).
  void insert(VID vid, const unsigned char *data, uint32_t length);

  // Drop one entry (a row's vector changed) or everything (the index changed
  // under us, or the budget was reduced).
  void invalidate(VID vid);
  void clear();

  // Change the budget, evicting down to it if it shrank.
  void set_max_bytes(size_t max_bytes);

  size_t resident_bytes() const { return m_bytes; }
  size_t entries() const { return m_map.size(); }
  uint64_t hits() const { return m_hits; }
  uint64_t misses() const { return m_misses; }

private:
  struct Node {
    VID vid;
    std::vector<unsigned char> bytes;
  };

  // MRU at the front. The map points at list nodes, whose addresses std::list
  // keeps stable across insert and erase of *other* elements -- which is what
  // makes the lifetime contract above hold.
  std::list<Node> m_lru;
  std::unordered_map<uint64_t, std::list<Node>::iterator> m_map;

  size_t m_max_bytes;
  size_t m_bytes = 0;
  uint64_t m_hits = 0;
  uint64_t m_misses = 0;

  void evict_to_fit(size_t incoming);
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
