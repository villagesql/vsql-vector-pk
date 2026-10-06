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

namespace svector::hnsw {

const quant::QData *VectorCache::get(VID vid) {
  auto it = m_map.find(vid.value);
  if (it == m_map.end()) {
    stat_add(vcache_misses, 1);
    return nullptr;
  }
  // Refresh: move to the front of the LRU. splice() relinks the node without
  // moving it, so the entry keeps its address and any pointer the caller still
  // holds stays valid.
  m_lru.splice(m_lru.begin(), m_lru, it->second);
  stat_add(vcache_hits, 1);
  return it->second->qdata();
}

const quant::QData *VectorCache::insert(VID vid, const float *src) {
  // Budget too small to hold even one entry: quantize into the scratch and hand
  // that back, so the caller still gets an operand. Nothing becomes resident.
  if (m_entry_bytes > m_max_bytes) {
    m_scratch.resize(m_entry_bytes);
    auto *q = reinterpret_cast<quant::QData *>(m_scratch.data());
    quant::quantize(src, m_dim, m_padded_dim, q);
    return q;
  }

  auto it = m_map.find(vid.value);
  if (it != m_map.end()) {
    // Already resident -- a re-fetch of an entry the caller re-read. Replace in
    // place rather than growing a duplicate.
    stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
    m_bytes -= m_entry_bytes;
    m_lru.erase(it->second);
    m_map.erase(it);
  }

  evict_to_fit(m_entry_bytes);

  m_lru.push_front(Node{vid, std::vector<unsigned char>(m_entry_bytes)});
  m_map.emplace(vid.value, m_lru.begin());
  m_bytes += m_entry_bytes;
  stat_add(vcache_resident_bytes, static_cast<long long>(m_entry_bytes));

  auto *q = m_lru.front().qdata();
  quant::quantize(src, m_dim, m_padded_dim, q);
  return q;
}

void VectorCache::invalidate(VID vid) {
  auto it = m_map.find(vid.value);
  if (it == m_map.end()) return;
  stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
  m_bytes -= m_entry_bytes;
  m_lru.erase(it->second);
  m_map.erase(it);
}

void VectorCache::clear() {
  stat_add(vcache_resident_bytes, -static_cast<long long>(m_bytes));
  m_lru.clear();
  m_map.clear();
  m_bytes = 0;
}

void VectorCache::set_max_bytes(size_t max_bytes) {
  m_max_bytes = max_bytes;
  evict_to_fit(0);
}

void VectorCache::evict_to_fit(size_t incoming) {
  while (m_bytes + incoming > m_max_bytes && !m_lru.empty()) {
    stat_add(vcache_resident_bytes, -static_cast<long long>(m_entry_bytes));
    stat_add(vcache_evictions, 1);
    m_bytes -= m_entry_bytes;
    m_map.erase(m_lru.back().vid.value);
    m_lru.pop_back();
  }
}

} // namespace svector::hnsw
