"""Client-side encoding micro-benchmark for SVECTOR ingest.

Measures only what the client does: turning a numpy float32 vector into the
text that goes into an INSERT/SELECT. No server, no network, no VEB.

Three encodings:
  text    '[f0,f1,...]'                     -- decimal literal
  binary  _binary X'<be16 0x8000|dim>
          <le f32...>'                     -- binary form (hex over PyMySQL's
                                               text protocol)
  raw     the bare bytes a binary-protocol
          driver would bind                 -- the MariaDB-parity floor
"""
import argparse
import struct
import time

import numpy


def enc_text(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def enc_binary(v):
    a = numpy.asarray(v, '<f4')
    header = struct.pack('>H', a.size | 0x8000)
    return "_binary X'" + header.hex() + a.tobytes().hex() + "'"


def enc_raw(v):
    return numpy.asarray(v, '<f4').tobytes()


ENCODINGS = [("text", enc_text, True), ("binary", enc_binary, True),
             ("raw", enc_raw, False)]


# Statement scaffolding, measured so the encoding cost is seen in context
# rather than in isolation. A value that goes into the SQL text is
# interpolated; raw bytes are bound as a parameter by a binary-protocol
# driver, so for those the statement text is built once with a placeholder
# and the bytes ride alongside it -- interpolating them would measure
# Python's repr() escaping, which no driver does.
def bench(fn, rows, build_stmt, inline):
    """Encode every row, optionally wrapping in a full INSERT statement."""
    t0 = time.perf_counter()
    total = 0
    if build_stmt and inline:
        for i, v in enumerate(rows):
            stmt = f"INSERT INTO t1 (id, v) VALUES ({i}, {fn(v)})"
            total += len(stmt)
    elif build_stmt:
        for i, v in enumerate(rows):
            stmt = f"INSERT INTO t1 (id, v) VALUES ({i}, %s)"
            params = (fn(v),)
            total += len(stmt) + len(params[0])
    else:
        for v in rows:
            total += len(fn(v))
    return time.perf_counter() - t0, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--rows", type=int, default=100_000)
    ap.add_argument("-d", "--dim", type=int, default=784)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--statement", action="store_true",
                    help="wrap each value in a full INSERT statement")
    args = ap.parse_args()

    print(f"{args.rows:,} vectors x {args.dim} dims, float32"
          f"{' (full INSERT statement)' if args.statement else ' (value only)'}")
    print(f"best of {args.repeat}\n")

    rng = numpy.random.default_rng(0)
    rows = rng.random((args.rows, args.dim), dtype=numpy.float32)

    # How often an untagged payload would be ambiguous: a bare element buffer
    # discriminated by "does it start with '['" misreads any vector whose
    # first element's low mantissa byte is 0x5B. The 0x8000 tag removes this
    # by construction -- the count is what the format has to defend against.
    first_bytes = rows[:, 0].astype('<f4').tobytes()[0::4]
    collisions = first_bytes.count(0x5B)

    results = []
    for name, fn, inline in ENCODINGS:
        best, payload = min(
            (bench(fn, rows, args.statement, inline)
             for _ in range(args.repeat)),
            key=lambda r: r[0])
        results.append((name, best, payload))

    base = results[0][1]
    base_bytes = results[0][2]
    w = max(len(n) for n, _, _ in results)
    print(f"{'':{w}}  {'total':>8}  {'per vec':>9}  {'speedup':>8}"
          f"  {'payload':>10}  {'vs text':>8}")
    for name, secs, payload in results:
        print(f"{name:{w}}  {secs:7.3f}s  {secs / args.rows * 1e6:8.1f}us"
              f"  {base / secs:7.1f}x  {payload / args.rows:9.0f}B"
              f"  {payload / base_bytes:7.2f}x")

    print(f"\nuntagged would be ambiguous for {collisions:,} of "
          f"{args.rows:,} vectors ({collisions / args.rows * 100:.2f}%): "
          f"first element byte is 0x5B ('[')")


if __name__ == "__main__":
    main()
