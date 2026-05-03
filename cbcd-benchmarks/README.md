# CBCD local benchmarks

This directory contains small local benchmark helpers for CBCD development runs.

Run the 1-rank orthogonal CBCD comparison from the repository root:

```bash
./cbcd-benchmarks/scripts/run_1rank_ortho.sh
```

The script writes timestamped logs and a compact `summary.txt` under
`cbcd-benchmarks/results/`. By default it runs old/new non-profiled cases and
old/new Caliper cases when the binaries expose Caliper. Pass `--no-caliper` to
skip Caliper, or `--caliper-scope new` to profile only the new binary.

Run repeated non-profiled comparisons with compact progress output:

```bash
./cbcd-benchmarks/scripts/run_1rank_ortho_repeats.sh --repeats 5 --jobs 1
```

Use `--jobs N` to run repeats concurrently. Keep `--jobs 1` for uncontended
single-GPU timing measurements. Each completed repeat prints elapsed time,
estimated remaining time, repeat duration, old/new sweep metrics, repeat
speedup, and running mean speedup.
