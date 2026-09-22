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

// ResidentGraph: an in-memory, pointer-based HNSW graph, materialized once from
// the on-disk index and then queried without touching storage. It is a SECOND
// backend for the same templated search algorithm (GraphOperations /
// LayerOperations) -- the on-disk IndexGraph is the other. The two share the
// search code; only the graph representation differs.
//
// Why: profiling the on-disk-with-cache path showed ~18% of query time in the
// vid->vector unordered_map::find (hash + bucket walk), co-equal with the
// distance math, plus ~6% in the visited-set hash. MHNSW pays ~0 for these
// because a neighbour IS a node pointer (node->vec is a field load, edges are
// pointer derefs). ResidentGraph replicates that: a node's neighbours are
// direct NodeObj* pointers and its quantized vector is stored inline, so
// distance() and neighbours() are pointer chases, not lookups.
//
// PoC scope (matches the insert-then-search benchmark flow): built once from
// the frozen on-disk graph at the first query, SINGLE-THREADED, no eviction, no
// write-invalidation. Locks are no-ops. Productionizing (concurrency, rebuild
// on write) is a later step.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_RESIDENT_GRAPH_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_RESIDENT_GRAPH_H

#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "../../quantize.h"
#include "hnsw.h"
#include "storage.h"

namespace svector::hnsw {

// One resident node: its ids (for result return), its per-level neighbour
// pointer lists, a visited generation stamp (cheaper than a hash-set), and its
// inline quantized vector. Allocated in an arena (ResidentGraph::m_nodes), so
// its address is stable and usable as a direct edge pointer.
// One resident node == one on-disk record (i.e. one node AT one level). The
// current storage gives a logical vector a distinct NID per level it occupies,
// chained by lower_level; we mirror that exactly (simplest, faithful): a
// NodeObj is a per-level record, and `lower` links a logical node's levels
// downward, just like the on-disk lower_level pointer. (When a future storage
// gives one NID per node, this collapses to one NodeObj per node.)
struct NodeObj {
  NID nid{};
  VID vid{};  // for the result path only (row resolution reads node->vid).
  bool deleted = false;

  // Resident pointer to this node's int16 quantized vector (packed contiguously
  // in ResidentGraph's vector arena). A resident pointer, so distance() is
  // node->vec (a deref), never a lookup.
  const quant::QData *vec = nullptr;

  // This node's neighbours AT THIS LEVEL, as DIRECT pointers to other resident
  // nodes -- following an edge is a dereference, no NID resolve, no fetch.
  std::vector<NodeObj *> neighbours;

  // The same logical node's record one level down (nullptr at level 0), mirror
  // of the on-disk lower_level pointer. Used by get_next_level_node.
  NodeObj *lower = nullptr;
};

class ResidentGraph {
public:
  using LevelId = LevelStore::LevelId;
  using DistanceType = double;

  // A search cursor over the resident graph: just a node pointer. key() (the
  // hashable identity the visited-set/dedup uses) is the pointer itself.
  struct Node {
    NodeObj *obj = nullptr;
    using KeyType = const NodeObj *;
    KeyType key() const { return obj; }
  };

  // The query operand: a pointer to the quantized query, held in the graph's
  // own m_query_qdata buffer (filled by prepare_query). Not a graph node.
  struct NodeData {
    const quant::QData *q = nullptr;
  };

  // Decodes + quantizes a raw encoded query vector ([ref][floats], as the
  // server hands it to begin()) into the graph's query buffer, returning it via
  // out. The resident graph owns all quantization -- storage passes the raw
  // vector through, unquantized. Returns true on error (err set).
  bool prepare_query(const IndexScanKey::KeyPartData &raw, NodeData &out,
                     char *err, uint32_t err_len);

  enum class LockMode { Shared, Exclusive };

  // Single-threaded PoC: the resident graph is frozen, so the locks are no-ops
  // that only satisfy the search algorithm's RAII lock-coupling contract.
  class LockGraph {
  public:
    LockGraph(ResidentGraph &, LockMode) {}
  };
  class LockLevels {
  public:
    enum class DescendPolicy { Keep, Release };
    LockLevels(ResidentGraph &g, LockMode, LevelId start, DescendPolicy)
        : m_level(start) {}
    LevelId descend() {
      if (m_level.has_lower_level()) m_level = m_level.lower();
      return m_level;
    }
    void update_policy(DescendPolicy) {}
    LevelId level() const { return m_level; }

  private:
    LevelId m_level;
  };

  ResidentGraph() = default;

  // The padded dim of the resident vectors (all uniform); used to size the SIMD
  // kernel's block loop. Set by the materializer.
  uint32_t padded_dim() const { return m_padded_dim; }

  // --- Read-side Graph interface (what GraphOperations::search_knn needs) ---

  bool get_entry_point(std::vector<Node> &out, LevelId &out_level) {
    out.clear();
    if (m_entry == nullptr) return false;
    out.push_back(Node{m_entry});
    out_level = LevelId{m_entry_level};
    return false;
  }

  // Follow the resident `lower` pointer (mirror of on-disk lower_level) to the
  // same logical node's record one level down.
  bool get_next_level_node(const Node &node, LevelId /*level*/, Node &out) {
    out = Node{node.obj->lower};
    return false;
  }

  bool visible(const Node &node, bool &out) {
    out = !node.obj->deleted;
    return false;
  }

  bool neighbours(const Node &node, LevelId /*level*/, std::vector<Node> &out) {
    // A NodeObj is a per-level record; its neighbours list IS this level's.
    out.clear();
    out.reserve(node.obj->neighbours.size());
    for (NodeObj *n : node.obj->neighbours) out.push_back(Node{n});
    return false;
  }

  // Fixed operand for is_dominated (matches the clean Graph contract): resolve
  // a node into a NodeData carrying its resident quantized vector pointer, so
  // the loop then calls distance(NodeData, Node) reusing it.
  bool resolve_fixed_operand(const Node &node, NodeData &out) {
    out.q = node.obj->vec;
    return false;
  }

  bool distance(const Node &a, const Node &b, DistanceType &out) {
    out = quant::dist_squared_l2_q(a.obj->vec, b.obj->vec, m_padded_dim);
    return false;
  }

  bool distance(const NodeData &a, const Node &b, DistanceType &out) {
    out = quant::dist_squared_l2_q(a.q, b.obj->vec, m_padded_dim);
    return false;
  }

private:
  friend class ResidentGraphBuilder;

  uint32_t m_padded_dim = 0;

  // Node arena: a deque so push_back keeps existing element addresses stable
  // (edges are NodeObj* pointers into it). Each NodeObj owns its per-level
  // neighbour pointer vectors directly.
  std::deque<NodeObj> m_nodes;

  // Vector arena: all nodes' int16 QData packed contiguously (one padded QData
  // per node). deque of byte-blocks keeps addresses stable; NodeObj::vec points
  // into it.
  std::deque<std::vector<unsigned char>> m_vector_storage;

  NodeObj *m_entry = nullptr;
  uint8_t m_entry_level = 0;

  // Owned buffers for prepare_query: the decoded f32 query, then its quantized
  // QData. Reused across queries (single-threaded PoC), so one search's query
  // does not allocate.
  std::vector<unsigned char> m_query_decoded;
  std::vector<unsigned char> m_query_qdata;
};

// Builds `out` from the frozen on-disk graph, driving reads through `src` (an
// IndexGraph over the same IndexStore). One-time, single-threaded. Returns true
// on error (err set). Defined in resident_graph.cc; declared here so begin()
// (storage.cc) can call it. Fwd-declared IndexGraph to avoid pulling graph.h
// into this header's includers.
class IndexGraph;
bool materialize_resident_graph(IndexGraph &src, const Index &index,
                                ResidentGraph &out, char *err,
                                uint32_t err_len);

}  // namespace svector::hnsw

#endif  // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_RESIDENT_GRAPH_H
