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

#include "resident_graph.h"

#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../native_vector.h"
#include "graph.h"

namespace svector::hnsw {

// Builds a ResidentGraph from the frozen on-disk graph. Runs once,
// single-threaded, at the first query (see begin()).
//
// NID-KEYED. A node's stable identity is its NID; the resident graph is keyed
// by NID and its edges are NIDs. The one place a vector is needed is behind a
// single conceptual operation, "vector for NID" (vector_for_nid): today that is
// implemented as nid -> (vid stored in the node's record) -> get vector by vid;
// in a future server-side design it becomes a nid -> (b-tree) -> vector lookup.
// The materializer depends only on that operation and on NID edges, never on
// the vid the current storage denormalizes into each edge -- so it survives the
// move to NID-only edges. The resident graph it produces has direct NodeObj*
// edges and a direct inline vector pointer: no lookup at query time.
//
// Algorithm:
//   1. get entry point + entry level.
//   2. For each level L from entry_level down to 0: BFS level L over the
//      on-disk neighbour NIDs. First time a NID is seen, create its NodeObj +
//      quantize its vector (vector_for_nid); record max_level. Stash each
//      node's neighbour NIDs for level L.
//   3. Descend the frontier one level (get_next_level_node) to seed level-1.
//   4. Wire every node's neighbours[L] as NodeObj* via the NID map.
class ResidentGraphBuilder {
public:
  ResidentGraphBuilder(IndexGraph &src, const Index &index, ResidentGraph &out,
                       char *err, uint32_t err_len)
      : m_src(src), m_index(index), m_out(out), m_err(err), m_err_len(err_len),
        m_max_col_len(index.get_max_col_len(VECTOR_KEY_POS)) {}

  bool build();

private:
  // Ensures a NodeObj exists for nid (creating + quantizing its vector on first
  // sight), returns it. Each nid is one per-level record, so no cross-level
  // merging -- a fresh nid always makes a fresh NodeObj. vid is kept on the
  // NodeObj only for the result path (row resolution reads node->vid).
  NodeObj *intern(NID nid, VID vid);

  // The single "vector for NID" operation the resident graph is built on.
  // Today: the vector id (vid) travels with the node, so this decodes the
  // vector by vid. Swap this body for a nid->b-tree->vector lookup when the
  // server provides one. Writes the quantized result into dst.
  bool vector_for_nid(NID nid, VID vid, quant::QData *dst);

  IndexGraph &m_src;
  const Index &m_index;
  ResidentGraph &m_out;
  char *m_err;
  uint32_t m_err_len;
  uint32_t m_max_col_len;

  uint32_t m_dim = 0;
  uint32_t m_padded_dim = 0;

  // NID value -> resident node (per-level record map, build-only).
  std::unordered_map<uint64_t, NodeObj *> m_by_nid;

  // Scratch for get_key_data (caller-provides-buffer) and the decoded f32.
  std::vector<unsigned char> m_raw;
  std::vector<unsigned char> m_decoded;
};

bool ResidentGraphBuilder::vector_for_nid(NID /*nid*/, VID vid,
                                          quant::QData *dst) {
  // --- "vector for NID", current implementation: nid's record carries its vid
  // (denormalized), so resolve the vector by that vid. Future: replace with a
  // server-side nid->b-tree->vector lookup; only this function changes. ---
  constexpr uint32_t kPrefix = IndexStore::KEY_REF_SIZE;
  if (m_raw.size() < m_max_col_len) m_raw.resize(m_max_col_len);
  IndexScanKey::KeyPartData raw;
  raw.data = m_raw.data();
  raw.length = m_max_col_len;
  if (m_index.get_key_data(VECTOR_KEY_POS,
                           static_cast<IndexScanKey::KeyPartRef>(vid.value),
                           &raw)) {
    snprintf(m_err, m_err_len, "ResidentGraph: get_key_data failed: %s",
             m_index.get_error());
    return true;
  }
  if (raw.length < kPrefix) {
    snprintf(m_err, m_err_len, "ResidentGraph: value too short (%u)",
             raw.length);
    return true;
  }
  if (native::from_encoded(raw.data + kPrefix, raw.length - kPrefix,
                           m_decoded.data(), m_decoded.size())) {
    snprintf(m_err, m_err_len, "ResidentGraph: decode failed");
    return true;
  }
  const auto *d = reinterpret_cast<const native::Data *>(m_decoded.data());
  quant::quantize(d->data, m_dim, m_padded_dim, dst);
  return false;
}

NodeObj *ResidentGraphBuilder::intern(NID nid, VID vid) {
  auto it = m_by_nid.find(nid.value);
  if (it != m_by_nid.end()) return it->second;

  // New per-level record: allocate its QData, quantize, create the NodeObj.
  m_out.m_vector_storage.emplace_back(quant::qdata_length(m_padded_dim));
  auto *q =
      reinterpret_cast<quant::QData *>(m_out.m_vector_storage.back().data());
  if (vector_for_nid(nid, vid, q)) return nullptr;

  m_out.m_nodes.emplace_back();
  NodeObj *n = &m_out.m_nodes.back();
  n->nid = nid;
  n->vid = vid;  // for the result path only (storage.cc row resolution).
  n->vec = q;
  m_by_nid.emplace(nid.value, n);
  return n;
}

bool ResidentGraphBuilder::build() {
  // Establish dim/padded_dim from the column, size the decode scratch.
  m_dim = (m_max_col_len - IndexStore::KEY_REF_SIZE) /
          static_cast<uint32_t>(sizeof(float));
  m_padded_dim = quant::qvector_padded_dim(m_dim);
  m_out.m_padded_dim = m_padded_dim;
  m_decoded.resize(native::length(m_dim).length);

  std::vector<IndexGraph::Node> entry;
  IndexGraph::LevelId entry_level{0};
  if (m_src.get_entry_point(entry, entry_level)) return true;
  if (entry.empty()) return false;  // empty index: resident graph stays empty

  // frontier = this level's seed nodes (on-disk Node, carry nid+vid).
  std::vector<IndexGraph::Node> frontier = entry;

  // Per level, remember (this-level NodeObj, its on-disk node) so that after
  // building the NEXT level down we can set each node's `lower` pointer.
  struct Descended {
    NodeObj *upper;         // the level-L NodeObj
    IndexGraph::Node down;  // its level-(L-1) on-disk node (nid+vid)
  };

  std::vector<Descended> pending_lower;

  for (int lv = entry_level.value; lv >= 0; --lv) {
    const IndexGraph::LevelId level{static_cast<uint8_t>(lv)};

    // BFS this level; every distinct level-L nid is its own record/NodeObj.
    std::unordered_set<uint64_t> seen;
    std::vector<IndexGraph::Node> queue = frontier;
    for (const auto &n : queue) seen.insert(n.nid.value);

    // First pass: intern every node reachable at this level (so all edge
    // targets exist before wiring). Collect per-node neighbour nids.
    std::vector<NodeObj *> level_objs;
    std::vector<std::vector<uint64_t>> level_edges;  // parallel to level_objs
    std::vector<IndexGraph::Node> nbr_buf;
    for (size_t i = 0; i < queue.size(); ++i) {
      const IndexGraph::Node cur = queue[i];
      NodeObj *obj = intern(cur.nid, cur.vid);
      if (obj == nullptr) return true;

      if (m_src.neighbours(cur, level, nbr_buf)) return true;
      std::vector<uint64_t> nids;
      nids.reserve(nbr_buf.size());
      for (const auto &nb : nbr_buf) {
        nids.push_back(nb.nid.value);
        if (seen.insert(nb.nid.value).second) queue.push_back(nb);
      }
      level_objs.push_back(obj);
      level_edges.push_back(std::move(nids));
    }

    // Wire this level's edges now that every level-L node is interned.
    for (size_t i = 0; i < level_objs.size(); ++i) {
      auto &dst = level_objs[i]->neighbours;
      dst.reserve(level_edges[i].size());
      for (uint64_t nid : level_edges[i]) {
        auto it = m_by_nid.find(nid);
        if (it != m_by_nid.end()) dst.push_back(it->second);
      }
    }

    // Resolve the `lower` links pending from the level ABOVE: their down-nodes
    // were just interned at this level.
    for (const auto &p : pending_lower) {
      auto it = m_by_nid.find(p.down.nid.value);
      if (it != m_by_nid.end()) p.upper->lower = it->second;
    }
    pending_lower.clear();

    if (lv == 0) break;

    // Descend every level-L node one level to (a) seed the next BFS and (b)
    // queue its `lower` link for resolution after level L-1 is built.
    std::vector<IndexGraph::Node> next_frontier;
    next_frontier.reserve(queue.size());
    for (size_t i = 0; i < queue.size(); ++i) {
      IndexGraph::Node down;
      if (m_src.get_next_level_node(queue[i], level, down)) return true;
      next_frontier.push_back(down);
      pending_lower.push_back({level_objs[i], down});
    }
    frontier = std::move(next_frontier);
  }

  m_out.m_entry = m_by_nid.at(entry.front().nid.value);
  m_out.m_entry_level = static_cast<uint8_t>(entry_level.value);
  return false;
}

bool ResidentGraph::prepare_query(const IndexScanKey::KeyPartData &raw,
                                  NodeData &out, char *err, uint32_t err_len) {
  // Storage hands us the raw encoded query ([ref][floats]); quantization is the
  // resident graph's job. Decode the floats, then quantize into the graph's own
  // query buffer -- same int16 form as the resident nodes.
  constexpr uint32_t kPrefix = IndexStore::KEY_REF_SIZE;
  if (raw.length < kPrefix) {
    snprintf(err, err_len, "ResidentGraph: query too short (%u)", raw.length);
    return true;
  }
  const uint32_t dim = (raw.length - kPrefix) / static_cast<uint32_t>(sizeof(float));
  if (m_query_decoded.size() < native::length(dim).length)
    m_query_decoded.resize(native::length(dim).length);
  if (native::from_encoded(raw.data + kPrefix, raw.length - kPrefix,
                           m_query_decoded.data(), m_query_decoded.size())) {
    snprintf(err, err_len, "ResidentGraph: failed to decode query vector");
    return true;
  }
  const auto *d = reinterpret_cast<const native::Data *>(m_query_decoded.data());

  if (m_query_qdata.size() < quant::qdata_length(m_padded_dim))
    m_query_qdata.resize(quant::qdata_length(m_padded_dim));
  auto *q = reinterpret_cast<quant::QData *>(m_query_qdata.data());
  quant::quantize(d->data, dim, m_padded_dim, q);
  out.q = q;
  return false;
}

bool materialize_resident_graph(IndexGraph &src, const Index &index,
                                ResidentGraph &out, char *err,
                                uint32_t err_len) {
  return ResidentGraphBuilder(src, index, out, err, err_len).build();
}

}  // namespace svector::hnsw

// --- Explicit template instantiation of the SEARCH path for ResidentGraph ---
// LayerOperations is read-only, so instantiate it fully. GraphOperations has
// mutation members ResidentGraph does not implement, so instantiate ONLY
// search_knn (and its private callees pulled in with it), NOT the whole class.
#include "graph_ops_impl.h"
#include "layer_ops_impl.h"
#include "visibility_policy.h"

namespace svector::hnsw {

template class LayerOperations<ResidentGraph, GraphVisiblePolicy>;
template class LayerOperations<ResidentGraph, AlwaysVisiblePolicy>;

template bool GraphOperations<ResidentGraph>::search_knn(
    const ResidentGraph::NodeData &, uint32_t, uint32_t,
    std::vector<GraphOperations<ResidentGraph>::Node> &);

}  // namespace svector::hnsw
