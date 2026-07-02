"""Shared modeling defaults."""

DEFAULT_RANDOM_SEED = 23

#: LightGBM thread-count for the direct train_model/train_models path (TUNE-6).
#: Single-threaded by default so a fixed seed reproduces bit-identical trees
#: regardless of machine core count -- the platform's "deterministic metrics"
#: invariant. Both this and DEFAULT_TUNE_NUM_THREADS live here so the two
#: training paths read the same single source instead of each hardcoding its
#: own literal inline.
DEFAULT_TRAIN_NUM_THREADS = 1

#: LightGBM thread-count for tune_hyperparameters' trial search (TUNE-6).
#: Historically 0 (LightGBM's "use every core") -- kept as the default since a
#: multi-hour, multi-recipe two-stage search benefits far more from wall-clock
#: parallelism than a single train_model call does; this means a tuned trial's
#: exact floating-point result is only reproducible on machines with the same
#: core count (LightGBM's documented determinism contract requires matching
#: num_threads), unlike the single-threaded direct-train path above. Set to 1
#: instead for cross-machine bit-identical trial reproduction, at the cost of
#: search wall-clock time.
DEFAULT_TUNE_NUM_THREADS = 0

__all__ = ["DEFAULT_RANDOM_SEED", "DEFAULT_TRAIN_NUM_THREADS", "DEFAULT_TUNE_NUM_THREADS"]
