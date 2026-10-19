# ADR 002 — Chunks are measured with one tokenizer and consumed by another

**Status:** accepted, with a known defect — 2026-09-10

## Context

The chunker sizes chunks with `tiktoken`/`cl100k_base`. The embedding model,
`bge-small-en-v1.5`, tokenises with WordPiece over a ~30k vocabulary and
truncates its input at 512 tokens (`truncation: {max_length: 512, direction:
right}`, read off the loaded tokenizer). The two counts are not the same number.

Measured over 305 chunks from three filings, comparing `cl100k_base` against a
copy of the model's own tokenizer with truncation disabled:

| | Median ratio WordPiece / cl100k |
|---|---|
| Prose | 1.03× |
| HTML tables | 1.44× |

| | |
|---|---|
| Chunks exceeding 512 WordPiece tokens | 78 of 305 — **25.6%** |
| Tokens never reaching a vector | 8,316 of 108,303 — **7.7%** |
| Worst single chunk | 492 cl100k → 773 WordPiece, 261 tokens dropped |

The six worst cases are all tables. HTML markup tokenises badly under a small
WordPiece vocabulary, so the loss falls hardest on exactly the content the
extraction strategy of ADR 001 was chosen to preserve.

The truncated text stays in the payload, so a retrieved chunk still shows it to
the model. It is retrieval that suffers: the tail of an oversized chunk is not
represented in its vector and therefore cannot be found.

## Decision

Change nothing for now. Ship the current chunker and let the week-3 evaluation
measure whether this costs recall.

The alternative considered and deferred: inject a `TokenCounter` port into the
chunker and count with the embedding model's own tokenizer, so `chunk_size`
means 512 tokens of the thing that actually consumes them. That removes the
defect entirely rather than bounding it, and remains the intended fix.

A cruder option — keeping tiktoken and dropping `chunk_size` to ~350 as a safety
margin — was rejected: it wastes about 30% of the budget on prose, whose ratio is
only 1.03, and a margin is still an approximation rather than a guarantee.

## Consequences

Retrieval on long tables is degraded by an amount not yet known. The golden
dataset must therefore include questions whose answer sits in the lower half of a
long table, otherwise the evaluation cannot see this defect at all and the
deferral proves nothing.

If `recall@5` on those questions comes out materially below the rest, the fix is
already identified and the measurement above is the justification for spending
the time on it.
