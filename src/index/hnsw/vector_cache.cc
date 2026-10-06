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


#include "vector_cache.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace svector::hnsw {

namespace {
// Index kept at <= 50% load so linear-probe runs stay short.
constexpr double kMaxLoad = 0.5;

uint64_t round_up_pow2(uint64_t v) {
  uint64_t p = 1;
  while (p < v) p <<= 1;
  return p;
}
}  // namespace

VectorCache::VectorCache(size_t max_bytes, uint32_t dim, uint32_t padded_dim)
    : m_max_bytes(max_bytes), m_dim(dim), m_padded_dim(padded_dim),
      m_entry_bytes(quant::qdata_length(padded_dim)) {
  m_capacity = static_cast<uint32_t>(m_max_bytes / m_entry_bytes);
  m_limit = m_capacity;
  if (m_capacity == 0) return;  // budget below one entry: insert() uses scratch

  // The slab is committed UP FRONT, for the whole budget, because that is what
  // makes slot addresses stable for the cache's life -- growing it later would
  // reallocate and invalidate every pointer a caller holds. So a budget is a
  // reservation, not a ceiling that is grown into: setting max_cache_size to
  // 512 MB costs 512 MB on the first query against the index, even if the
  // index only needs 96 MB.
  //
  // That is a deliberate trade (stable addresses are what let the kernel read
  // an entry in place, which is the whole point of this cache) but it makes
  // the default consequential. A future version could reserve the slab in
  // chunks and chain them, keeping addresses stable while allocating lazily.
  m_slab.resize(size_t{m_capacity} * m_entry_bytes);
  m_meta.resize(m_capacity);

  const uint64_t buckets = round_up_pow2(
      static_cast<uint64_t>(static_cast<double>(m_capacity) / kMaxLoad) + 1);
  m_index.assign(static_cast<size_t>(buckets), Bucket{});
  m_index_mask = buckets - 1;
}

VectorCache::Bucket *VectorCache::probe(uint64_t vid) {
  uint64_t i = mix(vid) & m_index_mask;
  Bucket *first_free = nullptr;
  for (;;) {
    Bucket &b = m_index[static_cast<size_t>(i)];
    if (b.vid == vid) return &b;
    // A tombstone does not end the run -- a key that collided past it is still
    // further along -- but it is claimable, so remember the first one and hand
    // it back if the key turns out to be absent.
    if (b.vid == kTombstone) {
      if (first_free == nullptr) first_free = &b;
    } else if (b.vid == 0) {
      return first_free != nullptr ? first_free : &b;
    }
    i = (i + 1) & m_index_mask;
  }
}

void VectorCache::index_insert(uint64_t vid, uint32_t slot) {
  Bucket *b = probe(vid);
  b->vid = vid;
  b->slot = slot;
}

void VectorCache::index_erase(uint64_t vid) {
  Bucket *b = probe(vid);
  if (b->vid != vid) return;
  // Leave a tombstone rather than reinserting the run that follows: relocating
  // a bucket is cheap, but repairing requires rewriting buckets a concurrent
  // probe may be walking. kTombstone keeps the run intact for lookups while
  // marking the slot gone; probe() treats it as occupied-but-not-matching.
  // Tombstones lengthen probe runs as they accumulate, which only a rebuild
  // clears -- acceptable because nothing erases during query-only work.
  b->vid = kTombstone;
}

const quant::QData *VectorCache::get(VID vid) {
  if (m_capacity == 0) {
    stat_add(vcache_misses, 1);
    return nullptr;
  }
  Bucket *b = probe(vid.value);
  if (b->vid != vid.value) {
    stat_add(vcache_misses, 1);
    return nullptr;
  }
  stat_add(vcache_hits, 1);
  return slot_data(b->slot);
}


const quant::QData *VectorCache::insert(VID vid, const float *src) {
  // Budget below one entry: nothing can ever become resident, so every insert
  // is a refusal.
  if (m_capacity == 0) {
    stat_add(vcache_admissions_refused, 1);
    return nullptr;
  }

  // Only ever reached after get() missed, so the vid must not be resident --
  // a second fill would rewrite a slot a caller may be reading.
  assert(probe(vid.value)->vid != vid.value);

  if (m_used == m_limit) {
    // Full. Nothing is evicted to make room: a slot is written once and never
    // rewritten, which is what keeps a QData* valid for the cache's life.
    // Report the refusal and let the caller quantize into its OWN buffer --
    // this cache has no per-caller storage to lend, and two operands are live
    // at once during a comparison.
    stat_add(vcache_admissions_refused, 1);
    return nullptr;
  }

  const uint32_t slot = m_used++;
  m_meta[slot].occupied = true;
  m_meta[slot].vid = vid.value;
  ++m_live;
  stat_add(vcache_resident_bytes, static_cast<long long>(m_entry_bytes));
  index_insert(vid.value, slot);

  auto *q = slot_data(slot);
  quant::quantize(src, m_dim, m_padded_dim, q);
  return q;
}

void VectorCache::invalidate(VID vid) {
  if (m_capacity == 0) return;
  Bucket *b = probe(vid.value);
  if (b->vid != vid.value) return;
  const uint32_t slot = b->slot;
  index_erase(vid.value);
  // The slot is retired, not recycled: its bytes stay untouched because a
  // caller may still be reading them, and a later fill takes a fresh slot. So
  // the resident set only ever shrinks until the cache is rebuilt.
  m_meta[slot].occupied = false;
  --m_live;
  stat_add(vcache_tombstones, 1);
  stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
}

void VectorCache::clear() {
  stat_add(vcache_resident_bytes,
           -static_cast<long long>(size_t{m_live} * m_entry_bytes));
  m_live = 0;
  m_used = 0;
  std::fill(m_meta.begin(), m_meta.end(), Meta{});
  std::fill(m_index.begin(), m_index.end(), Bucket{});
}

bool VectorCache::set_max_bytes(size_t max_bytes) {
  if (max_bytes == m_max_bytes) return false;  // called per scan; usually a no-op
  // A budget change is a rebuild, not a resize. The slab is sized once at
  // construction and its slots never move, so there is no in-place way to
  // honour a raise: the caller drops this cache and builds one at the new
  // size. Returning rather than resizing keeps that decision with the owner,
  // which is the only place that knows no reader holds an entry.
  stat_add(vcache_limit_changes, 1);
  vcache_limit_slots = static_cast<long long>(max_bytes / m_entry_bytes);
  return true;
}

} // namespace svector::hnsw
