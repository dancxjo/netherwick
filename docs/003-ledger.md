# 003 Ledger

The experience ledger is append-only and starts as JSONL in `data/ledger/YYYY-MM-DD/session.jsonl`. Every important transition is recorded so replay, training, and offline inspection all share one source of truth.
