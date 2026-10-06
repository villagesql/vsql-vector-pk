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

// VectorCache: a byte-bounded, VID-keyed cache of vectors in the form the
// distance kernel consumes -- decoded, quantized to int16, with the per-vector
// terms the metric needs already folded in.
//
// CONTRACT: this cache exists solely to drive graph traversal and candidate
// ranking inside the index. What it holds is a DERIVED, LOSSY, metric-specific
// form -- not the stored vector. It must never be read to answer anything the
// user sees; an entry is an operand for the kernel and nothing else.
//
// Why not raw bytes: a first version cached the page bytes as fetched, and made
// queries ~5% SLOWER despite a 99.98% hit rate. A hit replaced a (warm) page
// fetch with a hash lookup but left every per-call cost in place -- the decode
// still ran on every distance call, the copy into the evaluator's scratch still
// happened. Caching the kernel-ready form removes both, and halves the bytes so
// twice as many vectors fit a given budget.
//
// LIFETIME CONTRACT the caller depends on: get()/insert() return a pointer to
// an entry that the caller then reads in place -- resolve_node_data() hands it
// straight to the distance kernel, with no copy. That is the point of the
// design (the paged path must copy, because its bytes live under a page latch
// that has to be released before the distance loop runs; a cache entry has no
// latch, so it can be read where it lies). For that to be sound, an entry must
// not move or be freed while a caller holds a pointer into it. Today:
//
//   - std::list keeps node addresses stable across insert and erase of OTHER
//     elements, and get()'s splice() relinks without moving, so an entry's
//     address is stable for its whole residency.
//   - entries are evicted or invalidated only between queries, never during
//     one -- which holds because the index is single-threaded throughout, NOT
//     because anything enforces it.
//
// NOT THREAD-SAFE, and a mutex alone will not fix it. Blockers, worst first:
//
//   1. Eviction frees memory a live pointer still refers to. A caller holds
//      the QData* across the distance call; if another thread's insert()
//      evicts that entry, pop_back() frees it -- use-after-free. The pointer
//      escapes any lock held inside the cache, so locking the cache is not
//      sufficient. Fixes: pin on read (refcount/epoch, an atomic per access),
//      defer all eviction to query boundaries (cheap, and close to what the
//      contract above already claims -- but it has to be enforced in code
//      rather than asserted in a comment), or copy out and give up the
//      in-place read.
//   2. get() mutates the LRU: splice() relinks on every hit, so two concurrent
//      READERS corrupt the list. A read-only workload is not safe either.
//   3. IndexStore::vector_cache() creates and resizes lazily; two threads
//      racing there both construct and one leaks.
//
// Until those are addressed this is a single-threaded measurement vehicle, not
// a shippable cache.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "../../quantize.h"
#include "hnsw.h"
#include "stats.h"

namespace svector::hnsw {

class VectorCache {
public:
  // `padded_dim` is the quantized vector length every entry is built to (see
  // quant::qvector_padded_dim); the kernel requires it to be uniform, so it is
  // fixed for the cache's lifetime rather than carried per entry.
  VectorCache(size_t max_bytes, uint32_t dim, uint32_t padded_dim);

  // Look up vid. Returns the resident QData, or nullptr on a miss.
  const quant::QData *get(VID vid);

  // Quantize `dim` floats at `src` and make the result resident for vid,
  // returning it. Evicts if that would exceed the budget. Returns the entry
  // even when the budget is too small to retain it, so the caller always gets
  // an operand back.
  const quant::QData *insert(VID vid, const float *src);

  // Drop one entry (its VID was freed and may be reused) or everything.
  void invalidate(VID vid);
  void clear();

  // Change the budget. Shrinking evicts down to it; growing does not reallocate
  // the slab, so the new ceiling takes effect only up to the capacity the cache
  // was built with.
  void set_max_bytes(size_t max_bytes);

  uint32_t dim() const { return m_dim; }
  uint32_t padded_dim() const { return m_padded_dim; }
  size_t resident_bytes() const { return m_live * m_entry_bytes; }
  size_t entries() const { return m_live; }

private:
  // SLOT STORAGE. One slab, allocated once at capacity, entries at fixed
  // stride -- so a slot's address never moves and the lifetime contract above
  // holds by construction rather than by a container's guarantees. One
  // allocation for the whole cache, and a QData is reached by indexing rather
  // than by chasing a pointer.
  std::vector<unsigned char> m_slab;
  // Slots the slab was BUILT to hold. Fixed for the cache's life: the slab is
  // never reallocated, because that would move every entry and invalidate the
  // pointers callers hold.
  uint32_t m_capacity = 0;
  // Slots the CURRENT budget allows, <= m_capacity. Separate from capacity so
  // max_cache_size can be lowered at runtime: the slab keeps its allocation
  // but the cache holds fewer entries. Raising it again takes effect only up
  // to m_capacity.
  uint32_t m_limit = 0;
  uint32_t m_live = 0;      // slots currently occupied
  uint32_t m_used = 0;      // high-water mark: slots ever handed out
  // Slots freed by invalidate(). An invalidated slot is NOT compacted away by
  // moving another entry into it -- that would relocate bytes a caller may
  // still be pointing at -- so it is parked here and reused in place.
  std::vector<uint32_t> m_free;

  quant::QData *slot_data(uint32_t slot) {
    return reinterpret_cast<quant::QData *>(m_slab.data() +
                                            size_t{slot} * m_entry_bytes);
  }

  // CLOCK eviction, not LRU. An exact LRU relinks a list node on every hit --
  // six pointer writes across three scattered nodes -- and a profile put that
  // splice at 8.5% of query time while the ordering it maintained was never
  // read, because nothing had been evicted. CLOCK sets one byte in a cache
  // line the caller has just touched, and approximates LRU well enough: the
  // hand sweeps, clearing `referenced` on entries that have been used and
  // evicting the first that has not (second chance).
  struct Meta {
    uint64_t vid = 0;
    bool occupied = false;
    bool referenced = false;
  };
  std::vector<Meta> m_meta;
  uint32_t m_hand = 0;

  // OPEN-ADDRESSED INDEX, vid -> slot. Replaces std::unordered_map, whose
  // chaining costs two dependent cache misses per probe (bucket array, then
  // the chain node). This is one flat array of {vid, slot}: linear probing
  // keeps the common case to a single miss, and probes that do collide land on
  // the same cache line. Power-of-two sized and kept at <= 50% load so probe
  // runs stay short.
  //
  // A direct-mapped table would be better still, but a VID is a Column::Ref --
  // a {page, slot} encoding, not a sequence number -- so the key space is far
  // larger than the row count and sparse. Hence a hash, with a mixing step:
  // the low bits of a ref are slot indices that repeat across pages, and
  // libstdc++'s std::hash<uint64_t> is the identity, so the raw key clusters
  // badly.
  struct Bucket {
    uint64_t vid = 0;  // 0 == empty; VID::INVALID is 0 so this is unambiguous
    uint32_t slot = 0;
  };
  std::vector<Bucket> m_index;
  uint64_t m_index_mask = 0;

  static uint64_t mix(uint64_t x) {
    // splitmix64 finalizer: cheap, and spreads the low-entropy slot bits of a
    // Column::Ref across the whole word.
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }

  // Index probe: returns the bucket for vid, occupied or free to claim.
  Bucket *probe(uint64_t vid);
  void index_insert(uint64_t vid, uint32_t slot);
  void index_erase(uint64_t vid);
  // Linear probing leaves a hole on erase that would cut a probe run short, so
  // the run after the hole is reinserted.
  void index_repair_from(uint64_t start);

  // Evict one slot via the CLOCK hand. Returns the freed slot.
  uint32_t evict_one();

  // Scratch for an insert that cannot be retained (budget smaller than one
  // entry), so the caller still gets a usable operand.
  std::vector<unsigned char> m_scratch;

  size_t m_max_bytes;
  const uint32_t m_dim;
  const uint32_t m_padded_dim;
  const size_t m_entry_bytes;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
