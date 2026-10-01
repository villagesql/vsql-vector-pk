// Copyright (c) 2026 VillageSQL Contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2.0,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License, version 2.0, for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_PK_LAYOUT_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_PK_LAYOUT_H

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "hnsw.h"

namespace svector::hnsw {

// How a primary key is stored for row resolution, decided at index-create from
// the key's shape:
//   - inline: a single column whose max length fits PK_INLINE_MAX. The key
//   bytes
//     live directly in the level-0 node's fixed [pk_len:1][bytes:PK_INLINE_MAX]
//     field (pk_len <= PK_INLINE_MAX).
//   - spill: anything else (a larger single column, or a composite key). The
//   key
//     is packed into a separate fixed-width PK store; the node field holds a
//     reference to the packed record (pk_len == kSpillMarker).
// The packed spill record is, per key part in order: [len][bytes:part_max],
// where len is 1 byte if part_max <= 255 else 2 bytes, and bytes is zero-padded
// to part_max. Record width is the sum over all parts.
struct PkLayout {
  // pk_len sentinel in the node field meaning "spilled" (never a valid inline
  // length, which is <= PK_INLINE_MAX).
  static constexpr uint8_t kSpillMarker = 0xFF;

  // Upper bound on a spilled PK store record (Storage_spec::col_len, a
  // uint16_t). A composite or large key wider than this is rejected at index
  // create. Chosen well under the smallest supported data-page payload so the
  // record always fits a page alongside the page's own overhead.
  static constexpr uint32_t kMaxSpillRecordLen = 3072;

  bool inlined = false;           // true: store in the node field; false: spill
  uint32_t num_parts = 0;         // primary-key column count
  std::vector<uint32_t> part_max; // per-part max stored length
  uint32_t record_bytes = 0; // spill PK store record width (0 when inlined)

  static uint8_t len_prefix_bytes(uint32_t part_max) {
    return part_max <= 0xFF ? 1 : 2;
  }

  // Build the layout from the index's primary-key shape. record_bytes is the
  // full packed width even when it exceeds kMaxSpillRecordLen; the caller
  // (index create) rejects an over-large key via fits() before any storage is
  // created, so record_bytes is only consulted for a layout that fits.
  static PkLayout from_key(uint32_t num_parts, const uint32_t *part_max_lens) {
    PkLayout l;
    l.num_parts = num_parts;
    l.part_max.assign(part_max_lens, part_max_lens + num_parts);
    l.inlined = (num_parts == 1 && part_max_lens[0] <= PK_INLINE_MAX);
    if (!l.inlined) {
      uint32_t sum = 0;
      for (uint32_t i = 0; i < num_parts; ++i)
        sum += len_prefix_bytes(part_max_lens[i]) + part_max_lens[i];
      l.record_bytes = sum;
    }
    return l;
  }

  // Whether a spilled layout's packed record fits the PK store (and the
  // uint16_t Storage_spec::col_len). Always true for an inlined layout.
  bool fits() const { return inlined || record_bytes <= kMaxSpillRecordLen; }

  // Pack key parts into out (sized to record_bytes). parts[i] is the i'th
  // part's bytes+length (actual, <= part_max[i]).
  void
  pack(const std::vector<std::pair<const unsigned char *, uint32_t>> &parts,
       std::vector<unsigned char> &out) const {
    out.assign(record_bytes, 0);
    unsigned char *p = out.data();
    for (uint32_t i = 0; i < num_parts; ++i) {
      const uint32_t len = parts[i].second;
      if (len_prefix_bytes(part_max[i]) == 1) {
        *p++ = static_cast<unsigned char>(len);
      } else {
        *p++ = static_cast<unsigned char>((len >> 8) & 0xFF);
        *p++ = static_cast<unsigned char>(len & 0xFF);
      }
      if (len > 0)
        std::memcpy(p, parts[i].first, len);
      p += part_max[i];
    }
  }

  // Unpack a packed record into out_parts (each part's bytes copied out). rec
  // points at record_bytes bytes.
  void unpack(const unsigned char *rec,
              std::vector<std::vector<unsigned char>> &out_parts) const {
    out_parts.clear();
    out_parts.reserve(num_parts);
    const unsigned char *p = rec;
    for (uint32_t i = 0; i < num_parts; ++i) {
      uint32_t len;
      if (len_prefix_bytes(part_max[i]) == 1) {
        len = *p++;
      } else {
        len = (static_cast<uint32_t>(p[0]) << 8) | p[1];
        p += 2;
      }
      out_parts.emplace_back(p, p + len);
      p += part_max[i];
    }
  }
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_PK_LAYOUT_H
