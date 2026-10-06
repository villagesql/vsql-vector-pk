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

bool VectorCache::get(VID vid, Entry &out) {
  auto it = m_map.find(vid.value);
  if (it == m_map.end()) {
    ++m_misses;
    ++vcache_misses;
    return false;
  }
  // Refresh: move to the front of the LRU. splice() relinks the node without
  // moving it, so the bytes keep their address and any pointer the caller is
  // still holding stays valid.
  m_lru.splice(m_lru.begin(), m_lru, it->second);
  out.data = it->second->bytes.data();
  out.length = static_cast<uint32_t>(it->second->bytes.size());
  ++m_hits;
  ++vcache_hits;
  return true;
}

void VectorCache::insert(VID vid, const unsigned char *data, uint32_t length) {
  if (length == 0) return;
  // A vector that cannot fit on its own is never cached -- evicting the whole
  // cache to hold one entry would be worse than not caching it.
  if (length > m_max_bytes) return;

  auto it = m_map.find(vid.value);
  if (it != m_map.end()) {
    // Already resident (a concurrent-ish re-fetch, or a stale entry the caller
    // re-read): replace in place rather than growing a duplicate.
    vcache_resident_bytes -= static_cast<long long>(it->second->bytes.size());
    m_bytes -= it->second->bytes.size();
    m_lru.erase(it->second);
    m_map.erase(it);
  }

  evict_to_fit(length);

  m_lru.push_front(Node{vid, std::vector<unsigned char>(data, data + length)});
  m_map.emplace(vid.value, m_lru.begin());
  m_bytes += length;
  vcache_resident_bytes += static_cast<long long>(length);
}

void VectorCache::invalidate(VID vid) {
  auto it = m_map.find(vid.value);
  if (it == m_map.end()) return;
  vcache_resident_bytes -= static_cast<long long>(it->second->bytes.size());
  m_bytes -= it->second->bytes.size();
  m_lru.erase(it->second);
  m_map.erase(it);
}

void VectorCache::clear() {
  vcache_resident_bytes -= static_cast<long long>(m_bytes);
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
    auto &victim = m_lru.back();
    vcache_resident_bytes -= static_cast<long long>(victim.bytes.size());
    ++vcache_evictions;
    m_bytes -= victim.bytes.size();
    m_map.erase(victim.vid.value);
    m_lru.pop_back();
  }
}

} // namespace svector::hnsw
