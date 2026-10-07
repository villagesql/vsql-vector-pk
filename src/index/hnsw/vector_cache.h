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
// latch, so it can be read where it lies).
//
// FILL ONCE. A slot is written exactly once, when it is filled, and is never
// rewritten. Once the cache is full it stops admitting: a miss is quantized
// into scratch and handed back without becoming resident. Nothing is evicted
// to make room for anything else. So an entry's address AND its contents are
// stable for the life of the cache, and a QData* stays valid until the whole
// cache is torn down.
//
// Why not a replacement policy: measured on GCP at 60k x 784 (92 MB working
// set), an evicting cache held below that set runs at 0.66x of no cache at
// all -- the miss path, the quantize and the eviction are paid on top of the
// page read that happens anyway, for a hit rate too low to pay for them. Above
// the working set it never evicted at all. There is no budget at which
// replacement earned its cost, so the policy is to fill what fits and leave
// the rest to the paged path. Recency is also a poor predictor here: every
// query enters at the same top-layer points and fans out along different
// paths, so a leaf's recency says little about the next query.
//
// MIXED FORMS ARE DELIBERATE. A vector the cache refused is compared in f32,
// one it holds is compared in int16, within the same search. The alternative --
// quantizing a refused vector so every comparison matches -- measured WORSE
// than having no cache at all once the cache is full: the quantize is paid on
// every read and immediately thrown away, which at a quarter of the working
// set cost 3.6x on build and 0.41x on queries. Choosing the kernel per
// comparison instead brings that to 0.92x of no cache, degrading gracefully
// rather than falling off.
//
// What it gives up is that a distance no longer depends only on the two
// vectors: which kernel ran depends on what happened to be resident. int16
// error here is ~1e-4 relative, far below the gaps that decide neighbour
// ordering, and recall measured identical (1.0000) at every budget from a
// quarter of the working set to eight times it -- but two near-equidistant
// candidates could in principle order differently than they would have.
//
// Invalidate retires a slot rather than recycling it: the resident set only
// ever shrinks between rebuilds. It is not a delete path -- DELETE/UPDATE of a
// vector are unsupported -- its callers are VID-reuse guards on the write
// path, so query-only work never tombstones anything.
//
// CONCURRENCY. Readers run lock-free and write nothing. Fill-once is what
// makes that possible: a slot is written once, so there is no reclamation to
// protect against and a reader's QData* can never be invalidated under it --
// no pin, no epoch, no refcount on the read path.
//
// The protocol is two atomics:
//
//   - m_used is the allocation cursor. One fetch_add claims a slot, which is
//     what makes ownership exclusive without a lock. A claim past the limit is
//     given back and the insert is refused.
//   - Bucket::vid is the publication point. A filling thread writes the slot's
//     bytes and its index first, then RELEASES vid; a reader ACQUIRES vid
//     before trusting either. So a reader either does not see the key, and
//     takes the miss path, or sees it and is guaranteed finished bytes.
//
// invalidate() releases a tombstone the same way: a concurrent reader sees the
// key or does not, and both are correct answers. It is already serialized
// against other writers by the Level Operation Lock (see graph.h).
//
// Construction and replacement are serialized by a mutex in
// IndexStore::vector_cache(), taken once per scan -- never during traversal
// and never under a storage latch, so it sits outside the lock hierarchy
// rather than inside it.
//
// WHAT IS STILL NOT SAFE: nothing tracks which scans are using a cache, so a
// cache that is replaced is retired rather than freed (see
// retire_vector_cache()). A scan already in flight keeps using the old one,
// which is correct but means a budget or mode change leaks one cache until the
// store is destroyed. Bounded by operator actions, not by workload.
//
// The counters (hits, misses, resident bytes) are relaxed atomics, so they are
// exact in aggregate but may be read mid-update. entries() is likewise a
// snapshot. None of them is load-bearing for correctness.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H

#include <atomic>
#include <cstdint>
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
  // returning the entry. Returns NULLPTR when the cache is full: nothing is
  // evicted to make room, so the caller must quantize into its own buffer
  // instead. A refusal is not an error.
  const quant::QData *insert(VID vid, const float *src);

  // Drop one entry (its VID was freed and may be reused) or everything.
  void invalidate(VID vid);
  void clear();

  // Note a budget change. Returns true when the budget actually differs, which
  // means this cache must be REPLACED -- the slab is sized once at construction
  // and its slots never move, so a budget cannot be applied in place. The owner
  // drops the cache and builds one at the new size.
  bool set_max_bytes(size_t max_bytes);

  uint32_t dim() const { return m_dim; }
  uint32_t padded_dim() const { return m_padded_dim; }
  size_t resident_bytes() const {
    return m_live.load(std::memory_order_relaxed) * m_entry_bytes;
  }
  size_t entries() const { return m_live.load(std::memory_order_relaxed); }

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
  std::atomic<uint32_t> m_live{0};  // slots currently occupied
  // Slots ever handed out, and the ALLOCATION CURSOR: a filling thread claims
  // its slot with one fetch_add, which is what makes slot ownership exclusive
  // without a lock. Only ever grows, and never past m_limit -- a retired slot
  // is not returned for reuse, so this is a cursor rather than a high-water
  // mark.
  std::atomic<uint32_t> m_used{0};

  quant::QData *slot_data(uint32_t slot) {
    return reinterpret_cast<quant::QData *>(m_slab.data() +
                                            size_t{slot} * m_entry_bytes);
  }

  // Per-slot bookkeeping. No recency or reference bit: nothing is ever chosen
  // for replacement, so there is no ordering to maintain and get() stays a
  // pure read.
  struct Meta {
    uint64_t vid = 0;
    bool occupied = false;
  };
  std::vector<Meta> m_meta;

  // OPEN-ADDRESSED INDEX, vid -> slot. A chaining map (std::unordered_map)
  // costs two dependent cache misses per probe: the bucket array, then the
  // chain node. This is one flat array of {vid, slot}: linear probing
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
  // 0 == never used, kTombstone == retired. VID::INVALID is 0, so neither
  // sentinel can collide with a real key.
  static constexpr uint64_t kTombstone = ~uint64_t{0};
  // `vid` is the PUBLICATION POINT for an entry. A filling thread writes the
  // slot's bytes and its `slot` index first, then releases `vid` last; a
  // reader acquires `vid` before touching either. So a reader that sees a key
  // is guaranteed to see finished bytes behind it, and one that does not
  // simply takes the miss path. Nothing else about a bucket ever changes
  // after publication -- entries are written once -- so this one pair of
  // atomics is the whole protocol.
  struct Bucket {
    std::atomic<uint64_t> vid{0};
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


  size_t m_max_bytes;
  const uint32_t m_dim;
  const uint32_t m_padded_dim;
  const size_t m_entry_bytes;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_VECTOR_CACHE_H
