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


// Instrumentation counters, exposed as status variables (SHOW GLOBAL STATUS
// LIKE 'vsql_vector%') and registered in vector.cc.
//
// These are process-wide and aggregated across every index, which is enough to
// answer "how much work did that workload do" and "is the cache working".
// Per-index accounting would need the counters to hang off IndexStore and
// arrives, if ever, with the per-index cache settings.
//
// Plain long long rather than std::atomic, because the status variable
// capability registers a long long* that the server reads directly. Writers use
// relaxed atomic builtins so concurrent operations do not tear them; the
// server's read is an ordinary load. Relaxed is the right ordering: nothing
// inside the index ever makes a decision from these, so a lost update under
// contention costs a little accuracy and nothing else.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STATS_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STATS_H

#include <cstddef>

namespace svector::hnsw {

// --- search cost ---------------------------------------------------------
// Distance evaluations, and evaluate_distances() batches (one per expanded
// node or seed). The ratio is the mean fan-out actually walked; the first,
// divided by queries, is the work a given recall was bought with.
inline long long hnsw_distance_calls = 0;
inline long long hnsw_nodes_expanded = 0;

// --- vector cache --------------------------------------------------------
inline long long vcache_hits = 0;
inline long long vcache_misses = 0;
inline long long vcache_evictions = 0;
// A gauge, not a monotonic counter: bytes currently resident.
inline long long vcache_resident_bytes = 0;
// How the budget is being applied: how many times set_max_bytes() changed the
// limit, and the slot count it last resolved to. A budget that never reaches
// the cache leaves limit_changes at 0 while resident_bytes ignores it; a wrong
// limit shows up as an implausible limit_slots.
inline long long vcache_limit_changes = 0;
inline long long vcache_limit_slots = 0;

inline void stat_add(long long &counter, long long n) {
  __atomic_fetch_add(&counter, n, __ATOMIC_RELAXED);
}

inline void hnsw_count_distances(size_t n) {
  stat_add(hnsw_distance_calls, static_cast<long long>(n));
  stat_add(hnsw_nodes_expanded, 1);
}

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STATS_H
