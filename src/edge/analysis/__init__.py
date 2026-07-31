"""Phase-1 self-checks and read-only analysis scripts over the built feature tables.

Nothing here is imported by `edge.features` (INTERFACE.md's family modules stay pure); this
package only ever reads `data/features/*.parquet` (and, for the raw re-derivation in
`ofi_selfcheck.py`, `data/raw/lobster/*.csv` directly) and writes a report under `reports/`.
"""
