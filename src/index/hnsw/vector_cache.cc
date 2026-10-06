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
  for (;;) {
    Bucket &b = m_index[static_cast<size_t>(i)];
    if (b.vid == 0 || b.vid == vid) return &b;
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
  const uint64_t hole = static_cast<uint64_t>(b - m_index.data());
  b->vid = 0;
  index_repair_from(hole);
}

void VectorCache::index_repair_from(uint64_t hole) {
  // Walk the run after the hole, reinserting each entry. Without this a probe
  // for a key that collided past the hole would stop at it and report a miss.
  uint64_t i = (hole + 1) & m_index_mask;
  for (;;) {
    Bucket &b = m_index[static_cast<size_t>(i)];
    if (b.vid == 0) return;
    const uint64_t vid = b.vid;
    const uint32_t slot = b.slot;
    b.vid = 0;
    index_insert(vid, slot);
    i = (i + 1) & m_index_mask;
  }
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
  // CLOCK: one byte in the Meta the probe has already pulled in, rather than
  // an LRU list splice.
  m_meta[b->slot].referenced = true;
  stat_add(vcache_hits, 1);
  return slot_data(b->slot);
}

uint32_t VectorCache::evict_one() {
  // Second chance: clear `referenced` and move on; evict the first entry the
  // hand finds without it. An all-referenced sweep clears every bit on the
  // first pass and evicts on the second, so this terminates.
  assert(m_live > 0);
  for (;;) {
    Meta &m = m_meta[m_hand];
    const uint32_t slot = m_hand;
    m_hand = (m_hand + 1) % m_capacity;
    if (!m.occupied) continue;
    if (m.referenced) {
      m.referenced = false;
      continue;
    }
    index_erase(m.vid);
    m.occupied = false;
    --m_live;
    stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
    stat_add(vcache_evictions, 1);
    return slot;
  }
}

const quant::QData *VectorCache::insert(VID vid, const float *src) {
  // Budget below one entry: quantize into scratch so the caller still gets an
  // operand. Nothing becomes resident.
  if (m_capacity == 0) {
    m_scratch.resize(m_entry_bytes);
    auto *q = reinterpret_cast<quant::QData *>(m_scratch.data());
    quant::quantize(src, m_dim, m_padded_dim, q);
    return q;
  }

  uint32_t slot;
  Bucket *b = probe(vid.value);
  if (b->vid == vid.value) {
    slot = b->slot;  // already resident: overwrite in place
  } else if (!m_free.empty()) {
    slot = m_free.back();  // reuse a slot an invalidate freed
    m_free.pop_back();
    m_meta[slot].occupied = true;
    m_meta[slot].vid = vid.value;
    ++m_live;
    stat_add(vcache_resident_bytes, static_cast<long long>(m_entry_bytes));
    index_insert(vid.value, slot);
  } else if (m_used < m_limit) {
    slot = m_used++;  // slab not yet full: next never-used slot
    m_meta[slot].occupied = true;
    m_meta[slot].vid = vid.value;
    ++m_live;
    stat_add(vcache_resident_bytes, static_cast<long long>(m_entry_bytes));
    index_insert(vid.value, slot);
  } else {
    slot = evict_one();
    m_meta[slot].occupied = true;
    m_meta[slot].vid = vid.value;
    ++m_live;
    stat_add(vcache_resident_bytes, static_cast<long long>(m_entry_bytes));
    index_insert(vid.value, slot);
  }

  m_meta[slot].referenced = true;
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
  // The slot is NOT compacted away by moving another entry into it: that would
  // relocate that entry's bytes, and a caller may be holding a pointer to
  // them. Mark it free and let it be reused in place.
  m_meta[slot].occupied = false;
  m_meta[slot].referenced = false;
  m_free.push_back(slot);
  --m_live;
  stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
}

void VectorCache::clear() {
  stat_add(vcache_resident_bytes,
           -static_cast<long long>(size_t{m_live} * m_entry_bytes));
  m_live = 0;
  m_used = 0;
  m_hand = 0;
  m_free.clear();
  std::fill(m_meta.begin(), m_meta.end(), Meta{});
  std::fill(m_index.begin(), m_index.end(), Bucket{});
}

void VectorCache::set_max_bytes(size_t max_bytes) {
  if (max_bytes == m_max_bytes) return;  // called per scan; usually a no-op
  m_max_bytes = max_bytes;
  // The slab is not reallocated -- that would move every slot and invalidate
  // pointers callers hold -- so the budget moves m_limit within the capacity
  // already built. Lowering it evicts down to the new ceiling; raising it
  // takes effect only up to m_capacity.
  m_limit = std::min(static_cast<uint32_t>(m_max_bytes / m_entry_bytes),
                     m_capacity);
  stat_add(vcache_limit_changes, 1);
  vcache_limit_slots = m_limit;
  while (m_live > m_limit && m_live > 0) evict_one();
}

} // namespace svector::hnsw
