![VillageSQL Logo](https://villagesql.com/assets/logo-light.svg)

# VillageSQL Vector Extension (vsql-vector-pk)

An extension for VillageSQL Server that adds a vector data type with external
columnar storage (SVECTOR) and an HNSW index for nearest-neighbour search.

HNSW (Hierarchical Navigable Small World) is a graph index for approximate
k-nearest-neighbour (KNN) search: given a query vector it returns the k most
similar stored vectors without scanning the whole table.

This is `vsql-vector-pk`, a build of vsql-vector whose HNSW index stores each
row's primary key, so a KNN scan can resolve index hits back to full rows and
return their other columns — the plain vsql-vector index stores no row reference
and cannot do this. It builds and installs as the `vsql_vector` extension, and a
given server can run only one of vsql-vector and vsql-vector-pk at a time, since
both register under that same extension name.

> **This extension is under active development and is not stable.** It depends on
> [VillageSQL preview extension APIs](https://villagesql.com/docs/mysql-8.4/0.0.5-dev/extension-api-reference#preview-apis)
> that are subject to breaking changes without notice. It is not recommended for
> production use.

Primary keys may be a single column or a composite (multi-part) key, or the
table may have no primary key at all — InnoDB's synthetic row id is used in that
case. A small single-column key is held inline on the graph node; a larger or
composite key is packed into a separate per-index primary-key store. The only
limit is that a key whose declared maximum exceeds 3,072 packed bytes (including
the length bytes) is rejected: at `CREATE INDEX` with a clear error, and when the
index is declared inline at `CREATE TABLE` with the generic error 168.

## Features

- **SVECTOR Type**: A float32 vector type with declared dimension and external columnar storage (up to 3072 dimensions)
- **HNSW Index**: Graph-based KNN index with L2, L1, cosine, and inner-product metrics, queried through `ORDER BY <distance> ... LIMIT k`
- **Distance Functions**: L1 (Manhattan) and L2 (Euclidean) distance metrics
- **Similarity Functions**: Inner product, and angular distance via cosine distance
- **Utility Functions**: Norm computation, dimension query, hex dump, and formatted output
- **Native InnoDB Integration**: Columnar storage implemented via VillageSQL/InnoDB exposed storage APIs, inheriting ACID guarantees, MVCC, and buffer pool–based caching
- **Row resolution**: The HNSW index stores each row's primary key, so a KNN hit resolves back to its full row

## Installation

### Build from Source

#### Prerequisites
- VillageSQL build tree (specified via `VillageSQL_BUILD_DIR`)
- CMake 3.16 or higher
- C++20 compatible compiler

#### Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/villagesql/vsql-vector-pk.git
   cd vsql-vector-pk
   ```

2. Configure and build:
   ```bash
   mkdir -p build && cd build
   cmake .. -DVillageSQL_BUILD_DIR=/path/to/villagesql/build
   make -j8
   ```

   This produces `vsql_vector.veb` in the build directory. (`make -j$(nproc)`
   also works on Linux; macOS has no `nproc`, where `make -j` with no number
   runs unlimited parallel jobs — pass an explicit count instead.)

   The default build is optimized (`RelWithDebInfo`), and the distance kernels
   are always compiled at `-O3`. To build with debug symbols and assertions:
   ```bash
   cmake .. -DVillageSQL_BUILD_DIR=/path/to/villagesql/build -DWITH_DEBUG=ON
   ```

3. Install the VEB into the VillageSQL build tree (optional):
   ```bash
   make install
   ```

   This copies `vsql_vector.veb` to the `veb_output_directory` configured in the VillageSQL build tree, making it discoverable by the server without specifying a full path.

## Usage

### Loading the Extension

Because this extension depends on preview APIs, you must enable preview extensions before installing:

```sql
SET PERSIST vsql_allow_preview_extensions = ON;
INSTALL EXTENSION vsql_vector;
```

To uninstall:

```sql
UNINSTALL EXTENSION vsql_vector;
```

### SVECTOR Type

`SVECTOR(N)` declares a fixed length float32 vector column with dimension `N`. Every row in that column stores exactly `N` floats. The maximum supported dimension is 3072.

```sql
-- Create a table with a vector column
CREATE TABLE embeddings (
    id    INT PRIMARY KEY,
    vec   SVECTOR(4) NOT NULL
) ENGINE=InnoDB;

-- Add a vector column to an existing table
ALTER TABLE embeddings ADD COLUMN vec2 SVECTOR(4) NULL;

-- Insert vectors (number of elements must match the declared dimension)
INSERT INTO embeddings VALUES (1, '[1.0, 2.0, 3.0, 4.0]', NULL);  -- reference
INSERT INTO embeddings VALUES (2, '[1.0, 2.0, 3.0, 5.0]', NULL);  -- L1 dist from id=1: 1
INSERT INTO embeddings VALUES (3, '[1.0, 2.0, 5.0, 4.0]', NULL);  -- L1 dist from id=1: 2
INSERT INTO embeddings VALUES (4, '[2.0, 4.0, 6.0, 8.0]', NULL);  -- L1 dist from id=1: 10
INSERT INTO embeddings VALUES (5, '[9.0, 8.0, 7.0, 6.0]', NULL);  -- L1 dist from id=1: 20
```

### SQL Function Reference

#### Scalar / utility functions

| Function | Returns | Description |
|---|---|---|
| `VECTOR_DIMENSION(v)` | INT | Declared dimension of the vector |
| `VECTOR_MAX_DIMENSION()` | INT | Maximum supported dimension (3072) |
| `VECTOR_NORM(v)` | REAL | L2 (Euclidean) norm |
| `VECTOR_FORMAT(v, precision)` | STRING | Vector as a fixed-precision decimal string |
| `VECTOR_HEX(v)` | STRING | Raw float bytes as uppercase hex (useful for debugging) |

#### Distance and similarity functions

| Function | Returns | Description |
|---|---|---|
| `L1_DISTANCE(v1, v2)` | REAL | L1 (Manhattan) distance — sum of absolute differences |
| `L2_DISTANCE(v1, v2)` | REAL | L2 (Euclidean) distance — square root of sum of squared differences |
| `COSINE_DISTANCE(v1, v2)` | REAL | Cosine distance — `1 - cosine_similarity`; range [0, 2] |
| `INNER_PRODUCT(v1, v2)` | REAL | Dot product (similarity, not a metric); higher means more similar |

Both arguments to a distance/similarity function must have the same dimension. All functions return NULL if either argument is NULL.

### Example Queries

```sql
-- Norm of a stored vector
SELECT id, VECTOR_NORM(vec) AS norm FROM embeddings ORDER BY id;

-- L2 distance between two stored vectors
SELECT L2_DISTANCE(a.vec, b.vec) AS dist
FROM embeddings a, embeddings b
WHERE a.id = 1 AND b.id = 2;

-- Nearest-neighbour search by L1 distance, with the query vector given inline
-- as a string constant.
SELECT id, L1_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]') AS dist
FROM embeddings
ORDER BY dist ASC
LIMIT 2;
-- Expected result: id=1 dist=0, id=2 dist=1

-- Update a vector value (allowed only when the column has no HNSW index)
UPDATE embeddings SET vec = '[0.5, 0.5, 0.5, 0.5]' WHERE id = 1;

-- Delete a row containing a vector (allowed only when the column has no HNSW index)
DELETE FROM embeddings WHERE id = 2;
```

Updating a vector column and deleting a row are supported only while the column
has no HNSW index; see [Indexing (HNSW)](#indexing-hnsw) for the DML
restrictions that apply once an index exists.

### Indexing (HNSW)

An HNSW index for approximate KNN search is created on an `SVECTOR` column with
`USING EXTENDED(hnsw)`. The distance metric is selected by an operator class
after the column name (default is L2 if omitted):

| Operator class | Metric |
|---|---|
| `hnsw_l2` (default) | L2 (Euclidean) |
| `hnsw_l1` | L1 (Manhattan) |
| `hnsw_cosine` | Cosine |
| `hnsw_inner_product` | Inner product |

```sql
CREATE TABLE docs (
    id    INT PRIMARY KEY,
    vec   SVECTOR(4) NOT NULL,
    -- Inline: an L2 HNSW index with tuning parameters.
    INDEX idx_vec (vec hnsw_l2) USING EXTENDED(hnsw) WITH (M = 8, ef_construction = 64)
) ENGINE=InnoDB;

-- Or create the index separately (cosine metric, default parameters).
CREATE INDEX idx_vec_cos ON docs (vec hnsw_cosine) USING EXTENDED(hnsw);

-- Or via ALTER TABLE.
ALTER TABLE docs ADD INDEX idx_vec_ip (vec hnsw_inner_product) USING EXTENDED(hnsw);
```

The three statements above build three separate HNSW indexes on the same `vec`
column; adding more than one index to a column raises warning 1831 (duplicate
index) for each index after the first.

`WITH (M = ..., ef_construction = ...)` sets the HNSW build parameters. Query the
index with an `ORDER BY <distance>(col, <query vector>) LIMIT k`, using the
distance function matching the index's operator class:

```sql
SELECT id
FROM docs
ORDER BY L2_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]')
LIMIT 10;
```

The query vector must be a string literal, as shown. A placeholder (`?`) or a
user variable in its place fails with `ERROR 3219`.

#### Query-time tuning: `vsql_vector.ef_search`

`ef_search` is a session variable that sets the HNSW query-time search breadth
(the classic HNSW `ef`): the number of candidates the search keeps as it
traverses the graph. Higher values improve recall at the cost of speed; lower
values are faster but may miss nearest neighbours. It applies per connection to
subsequent KNN queries.

- Default: `40`. Range: `1` to `65536`.
- It is floored at the query's `LIMIT` — if `LIMIT k` exceeds `ef_search`, `k` is
  used, since the search must consider at least `k` candidates to return `k` rows.

```sql
-- Widen the search for higher recall on subsequent queries in this session.
SET SESSION vsql_vector.ef_search = 200;

SELECT id
FROM docs
ORDER BY L2_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]')
LIMIT 10;
```

#### Filtering with a `WHERE` clause (post-filter limitation)

A `WHERE` clause may be combined with a KNN `ORDER BY ... LIMIT`, but filtering
is applied after the index search, not before. The HNSW scan first returns its
nearest-neighbour candidates — at most `ef_search` of them, or the query's
`LIMIT` if that is larger (see the flooring rule above) — and the `WHERE`
predicate is then applied to that candidate set.

As a result, a selective `WHERE` can leave fewer than `LIMIT` rows, or even zero,
even when enough matching rows exist further out in the graph: rows that satisfy
the predicate but fall outside the top `ef_search` by distance are never
considered. Raising `vsql_vector.ef_search` widens the candidate pool and can
recover such rows, at the cost of a slower search; it is not a guarantee.

```sql
-- Assume docs also has an `id` filter column. The predicate is applied to the
-- top ef_search nearest rows only, so this may return fewer than 10 rows if few
-- of the nearest neighbours satisfy it. Raise ef_search to widen the pool.
SELECT id
FROM docs
WHERE id > 1000
ORDER BY L2_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]')
LIMIT 10;
```

#### Supported statements on an HNSW-indexed table

DDL and DML support on an HNSW-indexed table is currently limited. The index has
no B-tree, so most operations that would maintain its entries are rejected with a
clean error, while operations that do not touch it are allowed. Two cases are not
yet safe and are called out below.

- **INSERT** is supported; the index is maintained as rows are inserted. Note
  that a rolled-back or otherwise failed insert leaves the index damaged for now
  — the graph edges added for the row are not undone.
- **DELETE** of a row is not supported.
- **UPDATE** that changes the indexed vector column, or the primary key, is not
  supported.
- **UPDATE** of a column that is not indexed is supported — it changes no index
  ordering field, so the index is not touched.
- **`INSERT ... ON DUPLICATE KEY UPDATE`** is supported when a matching row
  exists and the update changes only columns that are not indexed. Only
  **`REPLACE`** is always refused on an existing row, because it resolves to a
  delete followed by an insert. When either inserts a brand-new row (no key
  conflict) it behaves as a plain, supported INSERT.
- **CREATE INDEX** can be declared at `CREATE TABLE`, or added to an empty table.
  Building an HNSW index over an already-populated table (`CREATE INDEX ... USING
  EXTENDED(hnsw)` on a table that already holds rows) is not safe yet and can
  crash the server.
- **CHECK TABLE** on an HNSW-indexed table is not safe yet and can crash the
  server.

## Testing

The extension includes tests using the MySQL Test Runner (MTR) framework.

### Running Tests

**Option 1 (Default): Using installed VEB**

```bash
cd /path/to/villagesql/build/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector-pk/mysql-test
```

**Option 2: Using a VEB from the build directory**

```bash
cd /path/to/villagesql/build/mysql-test
perl mysql-test-run.pl --suite=/path/to/vsql-vector-pk/mysql-test --veb-source-dir=/path/to/vsql-vector-pk/build
```

## Diagnostic Tools

The build also produces `svector_page_dump`, a standalone tool for inspecting
SVECTOR storage pages in InnoDB tablespace files. See
[`src/storage/tools/README.md`](src/storage/tools/README.md) for full usage and a
worked example of inspecting a data page with delete-marked records.

## Development

### Project Structure
```
vsql-vector-pk/
├── src/
│   ├── native_vector.h        # SVECTOR type definition and encode/decode
│   ├── native_vector.cc       # Encoding/decoding implementations
│   ├── distance_registry.h    # Distance/similarity function registry
│   ├── vector.cc              # VDF implementations and extension registration
│   ├── storage/               # Fixed-size column storage (InnoDB-backed)
│   │   ├── storage.h/.cc               # Column store engine (fixed-size records)
│   │   ├── root_page.h/.cc             # Root page structure and free-slot mgmt
│   │   ├── data_page.h/.cc             # Data page structure and slot mgmt
│   │   └── tools/                      # svector_page_dump and its parsers
│   │       ├── README.md               # Page dump tool documentation
│   │       ├── svector_page_dump.cc    # Page dump tool driver
│   │       ├── page_reader.h/.cc       # IBD file reader
│   │       ├── root_page_parser.h/.cc  # Root page parser
│   │       ├── data_page_parser.h/.cc  # Data page parser
│   │       ├── hnsw_layout.h/.cc       # HNSW record layout helpers
│   │       └── hnsw_graph.h/.cc        # HNSW graph rendering (-g)
│   └── index/
│       └── hnsw/              # HNSW index implementation
│           ├── hnsw.h                  # Shared HNSW types/constants
│           ├── pk_layout.h             # Primary-key inline/spill layout + packing
│           ├── storage.h/.cc           # Index storage (graph levels/nodes, PK store)
│           ├── graph.h/.cc             # Graph structure
│           ├── graph_ops.h/.cc         # Graph operations (search/insert)
│           ├── layer_ops.h/.cc         # Per-layer operations
│           ├── distance_evaluator.h    # Distance-kernel dispatch
│           └── visibility_policy.h     # MVCC visibility for scans
├── cmake/
│   └── FindVillageSQL.cmake  # CMake module to locate VillageSQL SDK
├── mysql-test/
│   ├── t/                    # MTR test files
│   └── r/                    # MTR expected results
├── unittest/                 # Standalone unit tests
├── manifest.json             # VEB package manifest
└── CMakeLists.txt            # Build configuration
```

## Reporting Bugs and Requesting Features

If you encounter a bug or have a feature request, please open an [issue](https://github.com/villagesql/vsql-vector-pk/issues) using GitHub Issues.

## License

License information can be found in the [LICENSE](./LICENSE) file.

## Contributing

VillageSQL welcomes contributions from the community. For more information, please see the [VillageSQL Contributing Guide](https://github.com/villagesql/villagesql-server/blob/main/CONTRIBUTING.md).

## Contact

- File a [bug or issue](https://github.com/villagesql/vsql-vector-pk/issues) and we will review
- Join the [Discord channel](https://discord.gg/KSr6whd3Fr)
