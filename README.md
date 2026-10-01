![VillageSQL Logo](https://villagesql.com/assets/logo-light.svg)

# VillageSQL Vector Extension (vsql-vector-pk)

An extension for VillageSQL Server that adds a vector data type with external columnar storage (SVECTOR), enabling efficient vector operations and laying the foundation for future ANN search and indexing.

This is `vsql-vector-pk`, a downstream build of vsql-vector whose HNSW index owns the owning row's primary key, so a KNN scan can resolve hits back to full rows (see the note below). It builds and installs as the `vsql_vector` extension.

> **This extension is under active development and is not stable.** It depends on [VillageSQL experimental extension APIs](https://villagesql.com/docs/mysql-8.4/0.0.5-dev/extension-api-reference#experimental-apis) that are subject to breaking changes without notice. It is not recommended for production use.

> **Downstream variant — HNSW index owns the row's primary key.** The HNSW index
> stores the owning row's primary key as part of its own on-disk structure, so an
> index hit resolves back to its **full row** and a KNN scan can return the row's
> other columns; mainline vsql-vector stores no row reference and cannot do this.
> This is the variant to use for end-to-end ANN.
>
> A small single-column key is held inline on the node; a larger or composite key
> is packed into a separate per-index primary-key store. Composite (multi-part)
> primary keys are fully supported, and a primary-key-less table is fine — InnoDB's
> synthetic row id is used. The only limit is that a key too large to pack
> (declared max over ~3 KB) is rejected at `CREATE INDEX`.

## Features

- **SVECTOR Type**: A float32 vector type with declared dimension and external columnar storage (up to 3072 dimensions)
- **Distance Functions**: L1 (Manhattan) and L2 (Euclidean) distance metrics
- **Similarity Functions**: Inner product, and angular distance via cosine distance
- **Utility Functions**: Norm computation, dimension query, hex dump, and formatted output
- **Native InnoDB Integration**: Columnar storage implemented via VillageSQL/InnoDB exposed storage APIs, inheriting ACID guarantees, MVCC, and buffer pool–based caching
- **Efficient Storage**: External columnar storage with heap-style organization (non-clustered), providing stable vector addresses suitable for ANN index structures
- **High Performance**: C++ implementation with bitmap-managed slot arrays and insert load distribution across pages via multiple free lists

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
   make -j$(nproc)
   ```

   This produces `vsql_vector.veb` in the build directory.

   To build with debug symbols and assertions (no optimization):
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

To unload:

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

-- Update a vector value
UPDATE embeddings SET vec = '[0.5, 0.5, 0.5, 0.5]' WHERE id = 1;

-- Delete a row containing a vector
DELETE FROM embeddings WHERE id = 2;
```

### Indexing (HNSW)

An approximate-nearest-neighbour (ANN) HNSW index is created on an `SVECTOR`
column with `USING EXTENDED(hnsw)`. The distance metric is selected by an operator
class after the column name (default is L2 if omitted):

| Operator class | Metric |
|---|---|
| `hnsw_l2` (default) | L2 (Euclidean) |
| `hnsw_l1` | L1 (Manhattan) |
| `hnsw_cosine` | Cosine |
| `hnsw_inner_product` | Inner product |

The HNSW index stores the owning row's primary key (see the note at the top), so
an index hit resolves back to its full row. The primary key may be a single
column or a **composite (multi-part) key**, or the table may be primary-key-less;
the only restriction is that a key too large to pack (declared max over ~3 KB) is
rejected at `CREATE INDEX`.

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

`WITH (M = ..., ef_construction = ...)` sets the HNSW build parameters. Query the
index with an `ORDER BY <distance>(col, <query vector>) LIMIT k`, using the
distance function matching the index's operator class:

```sql
SELECT id
FROM docs
ORDER BY L2_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]')
LIMIT 10;
```

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
is applied **after** the index search, not before. The HNSW scan first returns
its nearest-neighbour candidates — at most `ef_search` of them — and the `WHERE`
predicate is then applied to that candidate set.

As a result, a selective `WHERE` can leave **fewer than `LIMIT` rows, or even
zero**, even when enough matching rows exist further out in the graph: rows that
satisfy the predicate but fall outside the top `ef_search` by distance are never
considered. Raising `vsql_vector.ef_search` widens the candidate pool and can
recover such rows, at the cost of a slower search; it is not a guarantee.

```sql
-- The category filter is applied to the top ef_search nearest rows only, so
-- this may return fewer than 10 rows if few of the nearest neighbours are in
-- category 7. Raise ef_search to widen the pool.
SELECT id
FROM docs
WHERE category = 7
ORDER BY L2_DISTANCE(vec, '[1.0, 2.0, 3.0, 4.0]')
LIMIT 10;
```

DDL and DML support on an HNSW-indexed table is currently limited. The index has
no B-tree, so any operation that would maintain its entries is rejected with a
clean error (the server is never crashed); operations that do not touch it are
allowed:

- **CREATE INDEX** is supported on an empty or a **populated** table: `CREATE
  INDEX ... USING EXTENDED(hnsw)` builds the index over whatever rows already
  exist, and it can equally be declared at `CREATE TABLE` with data loaded
  afterwards.
- **INSERT** is supported; the index is maintained as rows are inserted.
- **DELETE** of a row is not supported.
- **UPDATE** that changes the indexed vector column, or the primary key, is not
  supported.
- **UPDATE** of a non-indexed (payload) column **is** supported — it changes no
  index ordering field, so the index is not touched.
- **Upserts** (`REPLACE`, `INSERT ... ON DUPLICATE KEY UPDATE`) are rejected when
  they hit an existing row, because they resolve to a delete or update. When they
  insert a brand-new row (no key conflict) they behave as a plain, supported
  INSERT.

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

The build also produces `svector_page_dump`, a standalone tool for inspecting SVECTOR storage pages in InnoDB tablespace files. See `src/storage/tools/README.md` for full usage.

### Example: inspecting a data page with delete-marked records

The following example shows output after inserting 5 rows into a `SVECTOR(4)` column and then deleting two of them (rows with id=2 and id=4) before the InnoDB purge thread has run. The delete-marked slots remain physically present in the page until purge. The SVECTOR column store holds the vector only; row identity lives in the HNSW index, not here.

```bash
$ svector_page_dump embeddings.ibd 4 -d 5 -r
```

```
IBD File: embeddings.ibd
Root Page Number: 4

SVECTOR Root Page
=================

Version:           1
Page Type:         1 (ROOT_PAGE)
Creator:           SVECTOR
Column Size:       16 bytes (4-dim float vector)

Data Pages:
  Total:           1
  Free:            1
  Head:            Page #5
  Tail:            Page #5

Free Slot Array:
  Max Capacity:    2048 slots
  Current Size:    1 slots
  Non-empty Slots: 1

================================================================================

SVECTOR Data Page
=================

Version:           1
Page Type:         2 (DATA_PAGE)
Free Slot Number:  0

SVECTOR Data Page Links:
  Previous:        Page #4294967295 (NULL)
  Next:            Page #4294967295 (NULL)

SVECTOR Free Page Links:
  Previous:        Page #4294967295 (NULL)
  Next:            Page #4294967295 (NULL)

Capacity:
  Max Records:     673
  Free Records:    668 (99.3%)
  Allocated:       5 (0.7%)
    Active:        3
    Deleted:       2

Record Bitmap:
  AADAD...................................
  ........................................
  (remaining 633 free slots omitted)
  (. = Free, A = Active, D = Deleted)

Records (showing from slot 0, up to 10 records):
  [  0] Trx ID:        1001 Data:[0.10, 0.20, 0.30, 0.40]
  [  1] Trx ID:        1002 Data:[0.90, 0.80, 0.70, 0.60] (DELETED)
  [  2] Trx ID:        1003 Data:[0.50, 0.50, 0.50, 0.50]
  [  3] Trx ID:        1004 Data:[3.14, 2.72, 1.41, 1.73] (DELETED)
  [  4] Trx ID:        1005 Data:[0.11, 0.22, 0.33, 0.44]
```

Each record holds the vector and its MVCC transaction id only. The owning row's
primary key is stored by the HNSW index (inline on a graph node or in the index's
primary-key store), not in this column store, so an index hit resolves back to
its row via the index.

Key observations:
- **`DELETED` records** (slots 1 and 3) are still physically present and visible to concurrent transactions that started before the DELETE committed (MVCC). They are reclaimed by the purge thread once no active transaction can see them.
- **Record Bitmap** encodes each slot's state in 2 bits: `A` = active (occupied, not deleted), `D` = delete-marked (occupied, pending purge), `.` = free (available for insert).
- **Free Slot Number `0`** means this data page is tracked at index 0 in the root page's free slot array, making it eligible for the next insert without a root page scan.
- **Max Records `673`** is the page capacity for `SVECTOR(4)` on a 16 KB InnoDB page, where each record is 24 bytes (16-byte vector + 8-byte MVCC trx id): `54 (header) + ⌈673×2/8⌉ (bitmap) + 673×24 (records) + 8 (page trailer) = 16383 bytes`.

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
│       └── hnsw/              # HNSW ANN index implementation
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
│   ├── r/                    # MTR expected results
│   └── unittest/             # Standalone unit tests
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
- Start a discussion in the project [discussions](https://github.com/villagesql/vsql-vector-pk/discussions)
- Join the [Discord channel](https://discord.gg/KSr6whd3Fr)
