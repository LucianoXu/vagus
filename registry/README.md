# Run registry

One JSON record per completed training run — the small, versioned side of
the archival split: bytes (checkpoints, tensorboard events, logs) live on
whichever machine's storage the `archive` block points to; this registry
carries everything needed to find, verify and reproduce them.

Record schema (informal):

- `run`: name, code commit, timing, SLURM job ids
- `recipe`: the resolved TrainConfig the run actually used
- `data`: identity of the token stream (source revision, tokenizer hash,
  totals) — copied from the run's meta.json
- `results`: final metrics
- `archive`: locations holding the bytes; the primary location contains a
  `SHA256SUMS` manifest covering every archived file, so any copy can be
  verified with `sha256sum -c SHA256SUMS`

Reproduction = clone the recorded commit, prepare data matching the
recorded manifest identity, run the embedded recipe.
