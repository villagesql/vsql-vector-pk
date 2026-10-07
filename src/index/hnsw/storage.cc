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

#include "storage.h"

#include "graph.h"
#include "graph_ops.h"

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace svector::hnsw {

using svector::RootPage;
using vsql::preview_storage::Error;

static_assert(IndexStore::KEY_REF_SIZE == 8, "KeyPartRef must be 8 bytes");

static void write_u64_be(std::string &buf, uint64_t v) {
  for (int i = 7; i >= 0; --i)
    buf.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

static uint64_t read_u64_be(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v = (v << 8) | static_cast<uint64_t>(p[i]);
  return v;
}

// Writes v's low 6 bytes big-endian at p and advances p past them -- the
// on-disk width of NID/VID (Id<Tag>::STORAGE_SIZE).
static inline void write_id48_be(std::byte *&p, uint64_t v) {
  for (int i = 5; i >= 0; --i)
    *p++ = static_cast<std::byte>((v >> (i * 8)) & 0xff);
}

// Reads a 6-byte big-endian id from p and advances p past it -- inverse of
// write_id48_be.
static inline uint64_t read_id48_be(const std::byte *&p) {
  uint64_t v = 0;
  for (int i = 0; i < 6; ++i)
    v = (v << 8) | static_cast<uint64_t>(*p++);
  return v;
}

void StorageMeta::encode(std::string *out) const {
  assert(name.size() <= UINT8_MAX);
  assert(entry_points.size() <= UINT8_MAX);

  out->clear();
  out->reserve(MIN_ENCODED_LEN + name.size() +
               entry_points.size() * ENTRY_POINT_LEN);
  out->push_back(static_cast<char>(VERSION));
  out->push_back(static_cast<char>(name.size()));
  out->append(name);
  out->push_back(static_cast<char>(level.value));
  out->push_back(static_cast<char>(entry_level.value));
  out->push_back(static_cast<char>(entry_points.size()));
  for (auto ref : entry_points)
    write_u64_be(*out, static_cast<uint64_t>(ref));
}

bool StorageMeta::decode(std::string_view data) {
  const auto *p = reinterpret_cast<const uint8_t *>(data.data());
  size_t rem = data.size();

  auto read1 = [&](uint8_t &v) -> bool {
    if (rem < 1)
      return true;
    v = *p++;
    --rem;
    return false;
  };

  uint8_t version;
  if (read1(version) || version != VERSION)
    return true;

  uint8_t name_len;
  if (read1(name_len) || rem < name_len)
    return true;
  name.assign(reinterpret_cast<const char *>(p), name_len);
  p += name_len;
  rem -= name_len;

  uint8_t lvl;
  if (read1(lvl))
    return true;
  level = LevelStore::LevelId{lvl};

  uint8_t el;
  if (read1(el))
    return true;
  entry_level = LevelStore::LevelId{el};

  uint8_t num_eps;
  if (read1(num_eps))
    return true;
  if (rem < static_cast<size_t>(num_eps) * ENTRY_POINT_LEN)
    return true;

  entry_points.resize(num_eps);
  for (uint8_t i = 0; i < num_eps; ++i) {
    entry_points[i] = static_cast<IndexScanKey::KeyPartRef>(read_u64_be(p));
    p += ENTRY_POINT_LEN;
    rem -= ENTRY_POINT_LEN;
  }
  return false;
}

static bool parse_u32(const char *str, uint32_t *out) {
  if (str == nullptr || *str == '\0' || *str == '-')
    return true;

  errno = 0;
  char *end = nullptr;
  unsigned long v = strtoul(str, &end, 10);

  if (end == str || *end != '\0' || errno == ERANGE ||
      v > std::numeric_limits<uint32_t>::max())
    return true;

  *out = static_cast<uint32_t>(v);
  return false;
}

bool Options::validate(char *error_msg, uint32_t error_msg_len) const {
  if (M < 2) {
    snprintf(error_msg, error_msg_len, "M must be at least 2, got %u", M);
    return true;
  }
  if (M > MAX_M) {
    snprintf(error_msg, error_msg_len, "M must not exceed %u", MAX_M);
    return true;
  }
  if (ef_construction < M) {
    snprintf(error_msg, error_msg_len, "ef_construction (%u) must be >= M (%u)",
             ef_construction, M);
    return true;
  }
  uint32_t max_ef_construction = M * MAX_EF_CONSTRUCTION_FACTOR;
  if (ef_construction > max_ef_construction) {
    snprintf(error_msg, error_msg_len,
             "ef_construction (%u) exceeds maximum allowed value (%u) for M=%u",
             ef_construction, max_ef_construction, M);
    return true;
  }
  return false;
}

bool Options::parse(const vef_index_param_t *params, uint32_t count,
                    Options *out, char *error_msg, uint32_t error_msg_len) {
  *out = Options{};
  bool seen_m = false;
  bool seen_ef_construction = false;

  for (uint32_t i = 0; i < count; ++i) {
    const char *key = params[i].key;
    const char *val = params[i].value;

    if (strcmp(key, "m") == 0) {
      if (seen_m) {
        snprintf(error_msg, error_msg_len, "duplicate option 'M'");
        return true;
      }
      if (parse_u32(val, &out->M)) {
        snprintf(error_msg, error_msg_len,
                 "M must be a positive integer, got '%s'", val);
        return true;
      }
      seen_m = true;

    } else if (strcmp(key, "ef_construction") == 0) {
      if (seen_ef_construction) {
        snprintf(error_msg, error_msg_len,
                 "duplicate option 'ef_construction'");
        return true;
      }
      if (parse_u32(val, &out->ef_construction)) {
        snprintf(error_msg, error_msg_len,
                 "ef_construction must be a positive integer, got '%s'", val);
        return true;
      }
      seen_ef_construction = true;

    } else {
      snprintf(error_msg, error_msg_len, "unknown option '%s'", key);
      return true;
    }
  }
  return out->validate(error_msg, error_msg_len);
}

uint16_t IndexStore::entry_len(LevelStore::LevelId level) const {
  auto max_neighbours = LevelStore::max_neighbours(level, m_num_neighbours);
  // Level 0 carries the inline primary key; upper levels do not.
  const bool has_pk = (level.value == 0);
  return static_cast<uint16_t>(NeighbourEntry::storage_size(
      max_neighbours, level.has_lower_level(), has_pk));
}

uint16_t IndexStore::overflow_len(LevelStore::LevelId level) const {
  auto capacity = LevelStore::overflow_capacity(level, m_num_neighbours);
  return static_cast<uint16_t>(OverflowEntry::storage_size(capacity));
}

bool LevelStore::insert(MtrCtx::Ref mtr, const NeighbourEntry &entry,
                        Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
                        char *err, uint32_t err_len) {
  const bool has_lower = m_level.has_lower_level();
  const bool has_pk = (m_level.value == 0);
  const uint32_t max_n = max_neighbours();
  assert(entry.neighbours.size() <= max_n);

  const size_t len = NeighbourEntry::storage_size(max_n, has_lower, has_pk);
  assert(buffer.size() >= len);

  std::byte *p = buffer.data();
  write_id48_be(p, entry.owner.value);
  if (has_lower)
    write_id48_be(p, entry.lower_level.value);

  for (uint32_t i = 0; i < max_n; ++i) {
    if (i < entry.neighbours.size()) {
      write_id48_be(p, entry.neighbours[i].nid.value);
      write_id48_be(p, entry.neighbours[i].vid.value);
    } else {
      write_id48_be(p, NID::INVALID);
      write_id48_be(p, VID::INVALID);
    }
  }
  write_id48_be(p, entry.overflow.value);

  // Primary key field (level 0 only): [pk_len:1][data:PK_INLINE_MAX], zero-
  // padded. pk_len <= PK_INLINE_MAX means the key is inline -- data holds that
  // many bytes from entry.pk_data. pk_len == PkLayout::kSpillMarker means the
  // key spilled -- data holds entry.pk_spill_ref (a Column::Ref) big-endian.
  // Written once here; never a partial (chunked) update.
  if (has_pk) {
    *p++ = static_cast<std::byte>(entry.pk_len);
    std::byte *field = p;
    for (size_t i = 0; i < PK_INLINE_MAX; ++i)
      field[i] = std::byte{0};
    if (entry.pk_len == PkLayout::kSpillMarker) {
      static_assert(sizeof(Column::Ref) <= PK_INLINE_MAX);
      uint64_t ref = static_cast<uint64_t>(entry.pk_spill_ref);
      for (int i = PK_INLINE_MAX - 1; i >= 0; --i) {
        field[i] = static_cast<std::byte>(ref & 0xFF);
        ref >>= 8;
      }
    } else {
      assert(entry.pk_len <= PK_INLINE_MAX);
      for (size_t i = 0; i < entry.pk_len; ++i)
        field[i] = static_cast<std::byte>(entry.pk_data[i]);
    }
    p += PK_INLINE_MAX;
  }
  assert(static_cast<size_t>(p - buffer.data()) == len);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  Column::Ref col_ref;
  if (m_store.insert(mtr, trx_ref, col_data, col_ref, err, err_len))
    return true;

  out = NID{static_cast<uint64_t>(col_ref)};
  return false;
}

bool LevelStore::insert(MtrCtx::Ref mtr, const OverflowEntry &entry,
                        Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
                        char *err, uint32_t err_len) {
  const uint32_t capacity = overflow_capacity();
  assert(entry.incoming.size() <= capacity);

  const size_t len = OverflowEntry::storage_size(capacity);
  assert(buffer.size() >= len);

  std::byte *p = buffer.data();
  for (uint32_t i = 0; i < capacity; ++i) {
    if (i < entry.incoming.size())
      write_id48_be(p, entry.incoming[i].value);
    else
      write_id48_be(p, NID::INVALID);
  }
  write_id48_be(p, entry.overflow.value);
  assert(static_cast<size_t>(p - buffer.data()) == len);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  Column::Ref col_ref;
  if (m_overflow.insert(mtr, trx_ref, col_data, col_ref, err, err_len))
    return true;

  out = NID{static_cast<uint64_t>(col_ref)};
  return false;
}

bool LevelStore::remove(MtrCtx::Ref mtr, StoreKind kind, NID nid,
                        Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  ColumnStore &store = (kind == StoreKind::Neighbour) ? m_store : m_overflow;
  return store.purge(mtr, trx_ref, nid.column_ref(), err, err_len);
}

bool LevelStore::mark_delete(MtrCtx::Ref mtr, NID nid, Segment::TrxRef trx_ref,
                             bool delete_mark, char *err, uint32_t err_len) {
  return m_store.mark_delete(mtr, trx_ref, nid.column_ref(), delete_mark, err,
                             err_len);
}

bool LevelStore::update(MtrCtx::Ref mtr, NID id, const NeighbourEntry &entry,
                        NodeField mask, const ScratchSlots &slots,
                        ScratchBytes &buffer, ScratchChunkIds &chunk_ids,
                        char *err, uint32_t err_len) {
  const bool has_lower = m_level.has_lower_level();
  const bool has_pk = (m_level.value == 0);
  const uint32_t max_n = max_neighbours();
  assert(!has(mask, NodeField::LowerLevel) || has_lower);

  const uint16_t overflow_idx = neighbour_overflow_chunk();
  // ColumnStore::update requires a full-record-length buffer (it writes only
  // the chunks named below). The record includes the inline pk tail on level 0,
  // so the buffer must span it even though update never writes the pk.
  const size_t len = NeighbourEntry::storage_size(max_n, has_lower, has_pk);
  assert(buffer.size() >= len);

  size_t num_chunks = 0;
  auto write_chunk = [&](uint16_t idx, uint64_t value) {
    std::byte *p = buffer.data() + static_cast<size_t>(idx) * CHUNK_SIZE;
    write_id48_be(p, value);
    assert(num_chunks < chunk_ids.size());
    chunk_ids[num_chunks++] = idx;
  };

  if (has(mask, NodeField::Owner))
    write_chunk(owner_chunk(), entry.owner.value);

  if (has(mask, NodeField::LowerLevel))
    write_chunk(lower_level_chunk(), entry.lower_level.value);

  if (has(mask, NodeField::Neighbours)) {
    const size_t n = entry.neighbours.size();
    assert(n <= slots.size());
    for (size_t i = 0; i < n; ++i) {
      assert(slots[i].is_valid());
      assert(slots[i].value < max_n);
      assert(i == 0 || slots[i].value > slots[i - 1].value);
      write_chunk(neighbour_nid_chunk(slots[i].value),
                  entry.neighbours[i].nid.value);
      write_chunk(neighbour_vid_chunk(slots[i].value),
                  entry.neighbours[i].vid.value);
    }
  }

  if (has(mask, NodeField::Overflow))
    write_chunk(overflow_idx, entry.overflow.value);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  return m_store.update(mtr, id.column_ref(), col_data,
                        chunk_ids.span(num_chunks), CHUNK_SIZE, err, err_len);
}

bool LevelStore::update(MtrCtx::Ref mtr, NID id, const OverflowEntry &entry,
                        OverflowField mask, const ScratchSlots &slots,
                        ScratchBytes &buffer, ScratchChunkIds &chunk_ids,
                        char *err, uint32_t err_len) {
  const uint32_t capacity = overflow_capacity();

  const uint16_t overflow_idx = overflow_chunk();
  const size_t len = (static_cast<size_t>(overflow_idx) + 1) * CHUNK_SIZE;
  assert(buffer.size() >= len);

  size_t num_chunks = 0;
  auto write_chunk = [&](uint16_t idx, uint64_t value) {
    std::byte *p = buffer.data() + static_cast<size_t>(idx) * CHUNK_SIZE;
    write_id48_be(p, value);
    assert(num_chunks < chunk_ids.size());
    chunk_ids[num_chunks++] = idx;
  };

  if (has(mask, OverflowField::Incoming)) {
    const size_t n = entry.incoming.size();
    assert(n <= slots.size());
    for (size_t i = 0; i < n; ++i) {
      assert(slots[i].is_valid());
      assert(slots[i].value < capacity);
      assert(i == 0 || slots[i].value > slots[i - 1].value);
      write_chunk(incoming_chunk(slots[i].value), entry.incoming[i].value);
    }
  }

  if (has(mask, OverflowField::Overflow))
    write_chunk(overflow_idx, entry.overflow.value);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  return m_overflow.update(mtr, id.column_ref(), col_data,
                           chunk_ids.span(num_chunks), CHUNK_SIZE, err,
                           err_len);
}

bool LevelStore::fetch(MtrCtx::Ref mtr, NID id, bool for_update,
                       NeighbourEntry &entry, size_t &num_valid, char *err,
                       uint32_t err_len, NodeField mask, IncomingFilter filter,
                       const ScratchChunkIds *slots) {
  Column::Data col_data;
  Column::Data rowid_prefix;
  Segment::TrxRef trx_ref;
  bool delete_marked = false;
  if (m_store.fetch(mtr, id.column_ref(), for_update, col_data, rowid_prefix,
                    trx_ref, delete_marked, err, err_len))
    return true;

  const auto *base = reinterpret_cast<const std::byte *>(col_data.data);
  auto read_chunk = [&](uint16_t idx) -> uint64_t {
    const std::byte *p = base + static_cast<size_t>(idx) * CHUNK_SIZE;
    return read_id48_be(p);
  };

  if (has(mask, NodeField::Owner))
    entry.owner = VID{read_chunk(owner_chunk())};

  if (has(mask, NodeField::LowerLevel) && m_level.has_lower_level())
    entry.lower_level = NID{read_chunk(lower_level_chunk())};

  if (has(mask, NodeField::Neighbours)) {
    // for_update: a later update() will write back by slot, so every
    // requested slot -- INVALID and filtered-out ones included -- is
    // written in place to keep entry.neighbours aligned with slots.
    // Read-only: no such alignment is needed, so INVALID and filtered-out
    // slots are dropped and the rest are compacted to the front.
    size_t write_idx = 0;
    size_t valid_count = 0;
    auto handle_slot = [&](uint16_t slot) {
      const uint64_t nid_val = read_chunk(neighbour_nid_chunk(slot));
      const NID nid{nid_val};
      const bool include =
          nid.is_valid() &&
          !(filter == IncomingFilter::ExcludeIncoming && nid.is_incoming());
      if (include) {
        ++valid_count;
      } else if (!for_update) {
        return;
      }
      entry.neighbours[write_idx++] =
          Node{nid, VID{read_chunk(neighbour_vid_chunk(slot))}};
    };

    if (slots != nullptr) {
      assert(slots->size() <= entry.neighbours.size());
      for (size_t i = 0; i < slots->size(); ++i)
        handle_slot((*slots)[i]);
    } else {
      const uint32_t max_n = max_neighbours();
      assert(max_n <= entry.neighbours.size());
      for (uint32_t slot = 0; slot < max_n; ++slot)
        handle_slot(static_cast<uint16_t>(slot));
    }

    if (!for_update)
      entry.neighbours = entry.neighbours.first(write_idx);
    num_valid = valid_count;
  }

  if (has(mask, NodeField::Overflow))
    entry.overflow = NID{read_chunk(neighbour_overflow_chunk())};

  // Primary key field (level 0 only), after the chunked region:
  // [pk_len:1][data:PK_INLINE_MAX]. For an inline key pk_data points at the key
  // bytes in-page (the caller copies before the mtr commits); for a spilled key
  // (pk_len == kSpillMarker) data holds the PK store ref, decoded big-endian
  // into pk_spill_ref.
  if (has(mask, NodeField::Pk) && m_level.value == 0) {
    const std::byte *tail =
        base + static_cast<size_t>(neighbour_overflow_chunk() + 1) * CHUNK_SIZE;
    entry.pk_len = static_cast<uint8_t>(tail[0]);
    const std::byte *field = tail + 1;
    entry.pk_data = reinterpret_cast<const unsigned char *>(field);
    if (entry.pk_len == PkLayout::kSpillMarker) {
      uint64_t ref = 0;
      for (size_t i = 0; i < PK_INLINE_MAX; ++i)
        ref = (ref << 8) | static_cast<uint8_t>(field[i]);
      entry.pk_spill_ref = static_cast<Column::Ref>(ref);
    }
  }

  return false;
}

bool LevelStore::fetch(MtrCtx::Ref mtr, NID id, bool for_update,
                       OverflowEntry &entry, size_t &num_valid, char *err,
                       uint32_t err_len, OverflowField mask,
                       const ScratchChunkIds *slots) {
  Column::Data col_data;
  Column::Data rowid_prefix;
  Segment::TrxRef trx_ref;
  bool delete_marked = false;
  if (m_overflow.fetch(mtr, id.column_ref(), for_update, col_data, rowid_prefix,
                       trx_ref, delete_marked, err, err_len))
    return true;

  const auto *base = reinterpret_cast<const std::byte *>(col_data.data);
  auto read_chunk = [&](uint16_t idx) -> uint64_t {
    const std::byte *p = base + static_cast<size_t>(idx) * CHUNK_SIZE;
    return read_id48_be(p);
  };

  if (has(mask, OverflowField::Incoming)) {
    // Same for_update/read-only distinction as the NeighbourEntry overload
    // above: preserve slot alignment for a later update(), or compact away
    // INVALID entries for a read-only lookup.
    size_t write_idx = 0;
    size_t valid_count = 0;
    auto handle_slot = [&](uint16_t slot) {
      const uint64_t nid_val = read_chunk(incoming_chunk(slot));
      const bool valid = NID{nid_val}.is_valid();
      if (valid) {
        ++valid_count;
      } else if (!for_update) {
        return;
      }
      entry.incoming[write_idx++] = NID{nid_val};
    };

    if (slots != nullptr) {
      assert(slots->size() <= entry.incoming.size());
      for (size_t i = 0; i < slots->size(); ++i)
        handle_slot((*slots)[i]);
    } else {
      const uint32_t capacity = overflow_capacity();
      assert(capacity <= entry.incoming.size());
      for (uint32_t slot = 0; slot < capacity; ++slot)
        handle_slot(static_cast<uint16_t>(slot));
    }

    if (!for_update)
      entry.incoming = entry.incoming.first(write_idx);
    num_valid = valid_count;
  }

  if (has(mask, OverflowField::Overflow))
    entry.overflow = NID{read_chunk(overflow_chunk())};

  return false;
}

bool LevelStore::resolve_owner(NID nid, Node &out, char *err,
                               uint32_t err_len) {
  NeighbourEntry entry;
  size_t num_valid;
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  bool failed = fetch(mtr, nid, /*for_update=*/false, entry, num_valid, err,
                      err_len, NodeField::Owner);
  mtr_ctx.commit();
  if (failed)
    return true;

  out = Node{nid, entry.owner};
  return false;
}

bool IndexStore::read_pkey(NID nid,
                           std::vector<std::vector<unsigned char>> &out_parts,
                           char *err, uint32_t err_len) {
  LevelStore *level0 = get_level(LevelStore::LevelId{0});
  if (level0 == nullptr) {
    snprintf(err, err_len, "HNSW: read_pkey: level 0 not loaded");
    return true;
  }

  // Read the node's pk field, then (for a spilled key) the PK store record it
  // references, both under one mtr. Holding the node record's page latch across
  // the PK store fetch keeps the pair atomic against a concurrent purge, which
  // frees the node and its PK record together: the latch bars the purge from
  // slipping in and freeing (and reusing) the PK record between the two reads.
  NeighbourEntry entry;
  size_t num_valid;
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  bool failed = level0->fetch(mtr, nid, /*for_update=*/false, entry, num_valid,
                              err, err_len, NodeField::Pk);
  if (!failed) {
    if (entry.pk_len == PkLayout::kSpillMarker) {
      ColumnStore &pk_store = m_multi_store.m_stores[S_PK_STORE_INDEX];
      Column::Data col_data;
      Column::Data rowid_prefix;
      Segment::TrxRef trx_ref;
      bool delete_marked = false;
      failed = pk_store.fetch(mtr, entry.pk_spill_ref, /*for_update=*/false,
                              col_data, rowid_prefix, trx_ref, delete_marked,
                              err, err_len);
      if (!failed)
        m_pk_layout.unpack(col_data.data, out_parts);
    } else {
      // pk_data points in-page; copy before the mtr commits.
      out_parts.assign(1, std::vector<unsigned char>(
                              entry.pk_data, entry.pk_data + entry.pk_len));
    }
  }
  mtr_ctx.commit();
  return failed;
}

bool IndexStore::write_spilled_pkey(Segment::TrxRef trx_ref,
                                    const std::vector<unsigned char> &packed,
                                    Column::Ref &out_ref, char *err,
                                    uint32_t err_len) {
  // Own mtr, released before the caller latches the node that will reference
  // this record: the insert path must never hold the PK store and node pages
  // latched at once, or it would latch them in the opposite order from the read
  // and purge paths (node then PK) and could deadlock. See create_node().
  ColumnStore &pk_store = m_multi_store.m_stores[S_PK_STORE_INDEX];
  Column::Data col_data{packed.data(), static_cast<uint32_t>(packed.size())};
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  bool failed = pk_store.insert(mtr, trx_ref, col_data, out_ref, err, err_len);
  mtr_ctx.commit();
  return failed;
}

bool IndexStore::purge_spilled_pkey(MtrCtx::Ref mtr, Column::Ref ref,
                                    Segment::TrxRef trx_ref, char *err,
                                    uint32_t err_len) {
  ColumnStore &pk_store = m_multi_store.m_stores[S_PK_STORE_INDEX];
  return pk_store.purge(mtr, trx_ref, ref, err, err_len);
}

LevelStore *IndexStore::locate(NID nid, StoreKind &kind, char *err,
                               uint32_t err_len) {
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  uint8_t root_idx = 0;
  bool failed = m_multi_store.get_root_index(mtr, nid.column_ref(), root_idx,
                                             err, err_len);
  mtr_ctx.commit();

  if (failed)
    return nullptr;

  kind = (root_idx & 1) ? StoreKind::Overflow : StoreKind::Neighbour;
  return get_level(LevelStore::LevelId{static_cast<uint8_t>(root_idx >> 1)});
}

LevelStore *IndexStore::ensure_levels(LevelStore::LevelId target, char *err,
                                      uint32_t err_len) {
  assert(target <= max_level());
  assert(m_levels[m_entry_level.value].has_value());

  // Levels above m_entry_level may already exist, wholly or in part: an
  // insert that created them and then failed leaves them behind, since
  // levels are never dropped. Every step below is therefore skipped when its
  // work is already done rather than asserted against. Re-running
  // init_root_page() for a half that exists would allocate a second root
  // page and overwrite the stored ref, orphaning the first page along with
  // any records already written into it.
  for (uint8_t l = m_entry_level.value + 1; l <= target.value; ++l) {
    RootId primary{LevelStore::LevelId{l}, RootId::Type::Primary};
    RootId overflow{LevelStore::LevelId{l}, RootId::Type::Overflow};

    // The two halves are created by separate calls, so a failure (or a crash
    // and restart) can leave a level with only its primary store: each half
    // is checked on its own, not inferred from the other.
    if (!m_multi_store.m_stores[root_index(primary)].initialized() &&
        m_multi_store.init_root_page(segment_index(primary),
                                     root_index(primary), err, err_len))
      return nullptr;
    if (!m_multi_store.m_stores[root_index(overflow)].initialized() &&
        m_multi_store.init_root_page(segment_index(overflow),
                                     root_index(overflow), err, err_len))
      return nullptr;

    if (!m_levels[l].has_value())
      m_levels[l].emplace(
          LevelStore::LevelId{l}, m_multi_store.m_stores[root_index(primary)],
          m_multi_store.m_stores[root_index(overflow)], m_num_neighbours);
  }
  return get_level(target);
}

bool IndexStore::set_entry_point(MtrCtx::Ref mtr, const Node &node,
                                 LevelStore::LevelId level, char *err,
                                 uint32_t err_len) {
  ColumnStore &primary = m_multi_store.m_stores[0];

  StorageMeta meta;
  if (meta.decode(primary.m_metadata)) {
    snprintf(err, err_len,
             "HNSW: set_entry_point: failed to decode level-0 metadata");
    return true;
  }
  meta.entry_level = level;
  meta.entry_points.assign(
      1, node.nid.is_valid()
             ? static_cast<IndexScanKey::KeyPartRef>(node.nid.value)
             : IndexScanKey::EMPTY_REF);

  std::string encoded;
  meta.encode(&encoded);
  if (primary.update_metadata(mtr, encoded, err, err_len))
    return true;

  m_entry_point = node;
  m_entry_level = level;
  return false;
}

uint8_t IndexStore::segment_index(const RootId &root) const {
  return root.level.value == 0 ? static_cast<uint8_t>(SegmentIndex::Primary)
                               : static_cast<uint8_t>(SegmentIndex::Secondary);
}

uint8_t IndexStore::root_index(const RootId &root) const {
  return root.level.value * 2 + (root.type == RootId::Type::Overflow ? 1 : 0);
}

void IndexStore::build_storage_specs(std::vector<Storage_spec> &specs) {
  specs.clear();
  specs.reserve(S_NUM_STORES);
  for (uint8_t l = 0; l < S_MAX_LEVEL; ++l) {
    LevelStore::LevelId level{l};
    std::string meta;

    // Level-0 primary store carries the global entry level and entry point.
    // Always encode one entry point (may be EMPTY_REF on an empty index).
    // All other stores use entry_level{0} and no entry points.
    std::vector<IndexScanKey::KeyPartRef> entry_pts;
    if (l == 0)
      entry_pts.push_back(
          m_entry_point.nid.is_valid()
              ? static_cast<IndexScanKey::KeyPartRef>(m_entry_point.nid.value)
              : IndexScanKey::EMPTY_REF);
    auto el = (l == 0) ? m_entry_level : LevelStore::LevelId{0};
    StorageMeta{("HNSW-L" + std::to_string(l)), level, el, entry_pts}.encode(
        &meta);
    specs.push_back({entry_len(level), std::move(meta)});

    StorageMeta{("HNSW-L" + std::to_string(l) + "-OV"),
                level,
                LevelStore::LevelId{0},
                {}}
        .encode(&meta);
    specs.push_back({overflow_len(level), std::move(meta)});
  }

  // When the primary key spills, a final fixed-width store (at
  // S_PK_STORE_INDEX) holds the packed keys. Its root page, like every
  // non-level-0 store, is allocated lazily on first use (IndexStore::create for
  // a fresh index, init_root_page below). Omitted entirely for an inlined key.
  if (!m_pk_layout.inlined) {
    assert(specs.size() == S_PK_STORE_INDEX);
    std::string meta;
    StorageMeta{"HNSW-PK", LevelStore::LevelId{0}, LevelStore::LevelId{0}, {}}
        .encode(&meta);
    assert(m_pk_layout.record_bytes <= PkLayout::kMaxSpillRecordLen);
    specs.push_back(
        {static_cast<uint16_t>(m_pk_layout.record_bytes), std::move(meta)});
  }
}

// The mode and the budget are read per call rather than captured, so a later
// move from a session variable to a per-index setting touches only this
// function. Creating the cache lazily also means an index queried only in
// CACHE_NONE mode never allocates one.
VectorCache *IndexStore::vector_cache(uint32_t dim) {
  // Serializes construction and replacement against other scans starting at
  // the same time: without it two threads both construct and one result is
  // leaked, or one frees a cache the other has just taken a pointer to. Held
  // only here, at the start of a scan -- never during traversal, and never
  // while a storage latch is held, so it sits outside the lock hierarchy in
  // graph.h rather than under it.
  //
  // It does NOT make the cache safe to drop while a scan is running; see
  // below.
  std::lock_guard<std::mutex> guard(m_vector_cache_mutex);
  const long long mode = read_cache_mode();
  // Only CACHE_VECTORS is implemented; the sysvar refuses anything higher
  // (MAX_IMPLEMENTED_CACHE_MODE), so this is the whole mapping today. When the
  // pinned-skeleton modes land they branch here.
  assert(mode <= MAX_IMPLEMENTED_CACHE_MODE);
  if (mode < CACHE_VECTORS) {
    // Mode turned off under a cache that already exists: retire it, so the
    // memory is returned rather than held for a mode nobody is using.
    retire_vector_cache();
    return nullptr;
  }
  const auto budget = static_cast<size_t>(read_max_cache_size());
  if (budget == 0) {
    retire_vector_cache();
    return nullptr;
  }
  // A budget change cannot be applied to a live cache -- slots are sized and
  // placed once -- so it is honoured by replacing the cache. Dropping it here
  // is safe for the same reason the rest of the cache is: this runs at the
  // start of a scan, before any entry pointer has been handed out.
  if (m_vector_cache != nullptr && m_vector_cache->set_max_bytes(budget))
    retire_vector_cache();
  if (m_vector_cache == nullptr)
    m_vector_cache = std::make_unique<VectorCache>(
        budget, dim, quant::qvector_padded_dim(dim));
  return m_vector_cache.get();
}

// A cache being replaced is not freed: a scan that started earlier still holds
// a raw pointer to it, and to entries inside it. Nothing tracks those readers,
// so the old cache is parked here and released only when the store itself is
// destroyed.
//
// That is a deliberate leak, bounded by how often the budget or the mode
// changes -- both operator actions, not workload events. The alternative is
// refcounting every scan's use of the cache, which puts an atomic on the one
// path this whole design exists to keep free of them. Revisit if a workload
// ever flips these settings often enough for the retained caches to matter.
void IndexStore::retire_vector_cache() {
  if (m_vector_cache != nullptr)
    m_retired_vector_caches.push_back(std::move(m_vector_cache));
}

void IndexStore::invalidate_vector_cache(VID vid) {
  if (m_vector_cache != nullptr) m_vector_cache->invalidate(vid);
}

void IndexStore::clear_vector_cache() {
  if (m_vector_cache != nullptr) m_vector_cache->clear();
}

bool IndexStore::create(const PkLayout &pk_layout, Space::Ref space_ref,
                        Segment::TrxRef trx_ref, const Options &opts, char *err,
                        uint32_t err_len) {
  m_pk_layout = pk_layout;
  m_num_neighbours = opts.M;
  m_ef_construction = opts.ef_construction;
  m_level_norm_factor = 1.0 / std::log(static_cast<double>(opts.M));

  std::vector<Storage_spec> specs;
  build_storage_specs(specs);

  // Formats only the primary (level-0 store) root page. Stores 1..63 are sized
  // and named but their root pages are INVALID_REF until the corresponding
  // level is created.
  if (m_multi_store.create(space_ref, trx_ref, specs, segment_total(), err,
                           err_len))
    return true;

  // Level-0 overflow root page: allocated from segment 0.
  RootId root{LevelStore::LevelId{0}, RootId::Type::Overflow};
  if (m_multi_store.init_root_page(segment_index(root), root_index(root), err,
                                   err_len))
    return true;

  // PK store root page (spilled keys only): allocated from segment 0 like the
  // rest of the level-0 stores. Created up front, not lazily, so an insert can
  // write a spilled key without having to format it first.
  if (!m_pk_layout.inlined &&
      m_multi_store.init_root_page(static_cast<uint8_t>(SegmentIndex::Primary),
                                   S_PK_STORE_INDEX, err, err_len))
    return true;

  m_levels[0].emplace(LevelStore::LevelId{0}, m_multi_store.m_stores[0],
                      m_multi_store.m_stores[1], opts.M);
  m_initialized = true;
  return false;
}

bool IndexStore::drop(Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  return m_multi_store.drop(trx_ref, err, err_len);
}

bool IndexStore::load(const PkLayout &pk_layout, Index::StorageRef storage_ref,
                      const Options &opts, char *err, uint32_t err_len) {
  m_pk_layout = pk_layout;
  m_num_neighbours = opts.M;
  m_ef_construction = opts.ef_construction;
  m_level_norm_factor = 1.0 / std::log(static_cast<double>(opts.M));

  std::vector<Storage_spec> specs;
  build_storage_specs(specs);

  if (m_multi_store.load(storage_ref, specs, err, err_len))
    return true;

  StorageMeta level0_meta;
  if (level0_meta.decode(m_multi_store.m_stores[0].m_metadata)) {
    snprintf(err, err_len, "HNSW: load: failed to decode level-0 metadata");
    return true;
  }
  assert(level0_meta.entry_points.size() <= 1);
  assert(level0_meta.level == LevelStore::LevelId{0});
  m_entry_level = level0_meta.entry_level;
  IndexScanKey::KeyPartRef entry_ref = level0_meta.entry_points.empty()
                                           ? IndexScanKey::EMPTY_REF
                                           : level0_meta.entry_points[0];
  NID entry_nid = (entry_ref != IndexScanKey::EMPTY_REF)
                      ? NID{static_cast<uint64_t>(entry_ref)}
                      : NID{};

  // Reconstruct each level whose stores have been created.
  auto num_root_pages = static_cast<uint8_t>(m_multi_store.m_stores.size());
  for (uint8_t lvl = 0; lvl < S_MAX_LEVEL; ++lvl) {
    uint8_t si = lvl * 2;
    if (si + 1 >= num_root_pages || !m_multi_store.m_stores[si].initialized() ||
        !m_multi_store.m_stores[si + 1].initialized())
      break;

#ifndef NDEBUG
    StorageMeta primary_meta, overflow_meta;
    assert(!primary_meta.decode(m_multi_store.m_stores[si].m_metadata));
    assert(!overflow_meta.decode(m_multi_store.m_stores[si + 1].m_metadata));
    assert(primary_meta.level == LevelStore::LevelId{lvl});
    assert(overflow_meta.level == LevelStore::LevelId{lvl});
#endif // NDEBUG
    m_levels[lvl].emplace(LevelStore::LevelId{lvl}, m_multi_store.m_stores[si],
                          m_multi_store.m_stores[si + 1], m_num_neighbours);
  }

  // Only the entry point's NID is persisted; resolve its VID once here so
  // entry_point() is a plain field read with no page access thereafter.
  m_entry_point = Node{};
  if (entry_nid.is_valid()) {
    LevelStore *store = get_level(m_entry_level);
    assert(store != nullptr);

    if (store->resolve_owner(entry_nid, m_entry_point, err, err_len))
      return true;
  }

  m_initialized = true;
  return false;
}

// The PkLayout for index's primary key: a single column that fits PK_INLINE_MAX
// is stored inline on the level-0 node, anything larger or composite is packed
// into the PK store. Derived identically at create and load, since it depends
// only on the key's fixed shape.
static PkLayout pk_layout_for(const Index &index) {
  const uint32_t num_parts = index.get_primary_num_key_cols();
  std::vector<uint32_t> part_max(num_parts);
  for (uint32_t i = 0; i < num_parts; ++i)
    part_max[i] = index.get_primary_max_col_len(i);
  return PkLayout::from_key(num_parts, part_max.data());
}

bool create(StorageCtx *ctx, const Index &index, Space::Ref space_ref,
            Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  // Decide how the owning row's primary key is stored, and reject a key too
  // large to spill here at CREATE INDEX rather than per-row at INSERT.
  PkLayout pk_layout = pk_layout_for(index);
  if (!pk_layout.fits()) {
    snprintf(err, err_len,
             "HNSW index primary key is too large: packed key is %u bytes, "
             "maximum is %u",
             pk_layout.record_bytes, PkLayout::kMaxSpillRecordLen);
    return true;
  }

  const auto *opts = index.options<Options>();
  assert(opts != nullptr);
  auto *store = ctx->user();
  if (store->create(pk_layout, space_ref, trx_ref, *opts, err, err_len))
    return true;
  ctx->set_ref(store->storage_ref());
  return false;
}

bool drop(StorageCtx *ctx, const Index & /*index*/, Segment::TrxRef trx_ref,
          char *err, uint32_t err_len) {
  return ctx->user()->drop(trx_ref, err, err_len);
}

bool load(StorageCtx *ctx, const Index &index, Index::StorageRef storage_ref,
          char *err, uint32_t err_len) {
  const auto *opts = index.options<Options>();
  assert(opts != nullptr);
  // The key shape is fixed, so the layout matches the one create() used --
  // including whether the PK store exists -- so the specs line up with what was
  // persisted.
  return ctx->user()->load(pk_layout_for(index), storage_ref, *opts, err,
                           err_len);
}

bool insert(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
            IndexScanKey::KeyPartData *key_columns,
            IndexScanKey::KeyPartData *pkey_columns,
            IndexScanKey::KeyPartRef *key_ref, char *err, uint32_t err_len) {
  IndexGraph graph(*ctx->user(), index, trx_ref,
                   index.get_max_col_len(VECTOR_KEY_POS),
                   std::span<char>(err, err_len));
  // Construction searches the graph far more than a query does, and reads the
  // same existing vectors through the same funnel, so it benefits from the
  // cache too. Safe because the cache is keyed by VID and this insert's own
  // vector gets a fresh one -- it cannot collide with an entry, and the
  // vectors it reads are not the ones it is writing.
  graph.set_vector_cache(ctx->user()->vector_cache(
      (index.get_max_col_len(VECTOR_KEY_POS) - IndexStore::KEY_REF_SIZE) /
      static_cast<uint32_t>(sizeof(float))));

  // HAS_ROW_REF: the server hands the owning row's primary key in pkey_columns
  // (one entry per key column). The index stores it on the level-0 node -- an
  // inline single column directly in the node, a larger or composite key in the
  // PK store -- to return at scan_fetch.
  IndexGraph::NodeData node_data{key_columns[VECTOR_KEY_POS]};
  node_data.pkey_parts = pkey_columns;
  node_data.num_pkey_parts = index.get_primary_num_key_cols();
  // The vector being inserted is not in the index yet, so the construction
  // search would otherwise compare it as f32 against quantized neighbours.
  // node_data.data must keep the ORIGINAL encoded bytes -- create_node() hands
  // them to the server to store -- so the quantized form rides alongside in
  // qdata rather than replacing it.
  if (graph.prepare_external_operand(node_data)) return true;
  Node top_node;
  if (GraphOperations<IndexGraph>(graph).insert(node_data, top_node))
    return true;

  // Belt and braces against a reused VID: purge() invalidates on the way out,
  // but a VID freed in a session that never ran a purge through this cache --
  // or freed before the cache existed -- would otherwise serve the previous
  // occupant's vector. Dropping the entry for a VID we have just written costs
  // one hash erase per insert.
  ctx->user()->invalidate_vector_cache(top_node.vid);

  *key_ref = static_cast<IndexScanKey::KeyPartRef>(top_node.nid.value);
  return false;
}

bool mark_delete(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
                 IndexScanKey::KeyPartRef *key_ref,
                 IndexScanKey::KeyPartData * /*key_columns*/,
                 IndexScanKey::KeyPartData * /*pkey_columns*/, bool delete_mark,
                 char *err, uint32_t err_len) {
  IndexStore *store = ctx->user();
  NID nid{static_cast<uint64_t>(*key_ref)};

  // key_ref always names the vector's top-level NeighbourEntry (see
  // insert()), so it always resolves to a Neighbour-kind record whose level
  // is that vector's top level -- exactly as in purge() below.
  StoreKind kind;
  LevelStore *level_store = store->locate(nid, kind, err, err_len);
  if (level_store == nullptr)
    return true;
  assert(kind == StoreKind::Neighbour);

  Node target;
  if (level_store->resolve_owner(nid, target, err, err_len))
    return true;

  // The mark belongs on every level's record for this vector, not just the
  // top one key_ref names, so it goes through the graph operation that walks
  // the whole per-level chain.
  IndexGraph graph(*store, index, trx_ref,
                   index.get_max_col_len(VECTOR_KEY_POS),
                   std::span<char>(err, err_len));
  return GraphOperations<IndexGraph>(graph).mark_delete(
      target, level_store->level(), delete_mark);
}

bool purge(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
           IndexScanKey::KeyPartRef *key_ref,
           IndexScanKey::KeyPartData * /*key_columns*/,
           IndexScanKey::KeyPartData * /*pkey_columns*/, char *err,
           uint32_t err_len) {
  IndexStore *store = ctx->user();
  NID nid{static_cast<uint64_t>(*key_ref)};

  // key_ref is always the NID insert() returned -- the vector's own
  // top-level NeighbourEntry (see storage.cc's insert()) -- so it locates a
  // Neighbour-kind record whose level is that vector's top level.
  StoreKind kind;
  LevelStore *level_store = store->locate(nid, kind, err, err_len);
  if (level_store == nullptr)
    return true;
  assert(kind == StoreKind::Neighbour);

  Node target;
  if (level_store->resolve_owner(nid, target, err, err_len))
    return true;

  // The VID is about to be freed and can be handed to a later insert, which
  // would otherwise be served this vector's bytes from the cache.
  store->invalidate_vector_cache(target.vid);

  IndexGraph graph(*store, index, trx_ref,
                   index.get_max_col_len(VECTOR_KEY_POS),
                   std::span<char>(err, err_len));
  return GraphOperations<IndexGraph>(graph).remove(target,
                                                   level_store->level());
}

bool begin(StorageCtx *ctx, const Index &index, MtrCtx::Ref /*mctx*/,
           const IndexScanDesc &scan_desc, Index::Cursor *cursor, bool *eof,
           char *err, uint32_t err_len) {
  // The only capability this index registers is KNN (see vector.cc), so the
  // server never issues a Point/Range scan against it.
  assert(scan_desc.is_knn());
  assert(scan_desc.num_keys() == 1 && scan_desc[0].is_knn());

  IndexGraph graph(*ctx->user(), index, Segment::TrxRef{},
                   index.get_max_col_len(VECTOR_KEY_POS),
                   std::span<char>(err, err_len));
  // Cache is owned by the store, so it outlives this per-scan graph and the
  // next query sees what this one loaded. Null in CACHE_NONE.
  graph.set_vector_cache(ctx->user()->vector_cache(
      (index.get_max_col_len(VECTOR_KEY_POS) - IndexStore::KEY_REF_SIZE) /
      static_cast<uint32_t>(sizeof(float))));

  IndexGraph::NodeData query{scan_desc[0][VECTOR_KEY_POS]};
  // The query vector never passes through the cache -- it is not in the index
  // -- so quantize it here to match the operands it is about to be compared
  // against. Once per search.
  if (graph.prepare_external_operand(query)) return true;
  const uint32_t k = scan_desc.limit();
  const uint32_t ef_search =
      std::max<uint32_t>(k, static_cast<uint32_t>(read_ef_search()));

  // Materialize the FULL ef_search-ranked pool into the cursor, not just the
  // top k. The search already ranks all ef_search candidates (k is only a
  // post-hoc truncation), so returning the extra (ef_search - k) costs nothing
  // but retaining their Node refs in the cursor (~16 B each; bounded by the
  // thread pool). The server pulls them nearest-first and resolves each to a
  // row only on demand, so the tail is free unless it is actually consumed.
  // This gives the server local backfill for two post-index filters without an
  // immediate index round-trip:
  //   - a WHERE clause that eliminates some of the top k, and
  //   - MVCC visibility: a hit on a concurrently-inserted (invisible) row makes
  //     the clustered lookup return DB_RECORD_NOT_FOUND; the server skips it and
  //     pulls the next candidate (see custom_index_knn_scan.cc Read()).
  // Passing ef_search as the result count makes search_knn keep the whole pool.
  std::vector<Node> nodes;
  if (GraphOperations<IndexGraph>(graph).search_knn(query, ef_search, ef_search,
                                                    nodes))
    return true;

  auto *c = new Cursor(ctx->user(), std::move(nodes));
  *cursor = c;
  *eof = c->eof();
  return false;
}

bool position(Index::Cursor cursor, Index::CursorOp op, bool *eof,
              char * /*err*/, uint32_t /*err_len*/) {
  *eof = static_cast<Cursor *>(cursor)->advance(op);
  return false;
}

bool fetch(Index::Cursor cursor, Column::Ref *col_refs,
           IndexScanKey::KeyPartData * /*key_columns*/,
           IndexScanKey::KeyPartData *pkey_columns, char *err,
           uint32_t err_len) {
  auto *c = static_cast<Cursor *>(cursor);
  const Node *node = c->current();
  assert(node != nullptr);

  // vid is the vector's own stable column reference.
  col_refs[VECTOR_KEY_POS] = static_cast<Column::Ref>(node->vid.value);

  // Return the row identity: the primary key stored on this node's level-0
  // record (inline, or unpacked from the PK store), so the server resolves the
  // row directly from pkey_columns. Held in the cursor so each {data,len} stays
  // valid after this callback returns.
  auto &pkey = c->current_pkey();
  if (c->store()->read_pkey(node->nid, pkey, err, err_len))
    return true;
  for (size_t i = 0; i < pkey.size(); ++i)
    pkey_columns[i] = IndexScanKey::KeyPartData{
        pkey[i].data(), static_cast<uint32_t>(pkey[i].size())};
  return false;
}

bool save(Index::Cursor /*cursor*/, char * /*err*/, uint32_t /*err_len*/) {
  // The cursor holds no page latches or mtr state of its own -- the result
  // set was fully materialized in begin() -- so there's nothing to detach.
  return false;
}

bool restore(Index::Cursor cursor, MtrCtx::Ref /*mctx*/, bool *eof,
             char * /*err*/, uint32_t /*err_len*/) {
  *eof = static_cast<Cursor *>(cursor)->eof();
  return false;
}

void end(Index::Cursor *cursor) {
  delete static_cast<Cursor *>(*cursor);
  *cursor = nullptr;
}

} // namespace svector::hnsw
