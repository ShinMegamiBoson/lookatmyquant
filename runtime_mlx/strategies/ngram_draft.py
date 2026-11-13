"""Skip the neural drafter when a repeated suffix has an agreed continuation."""

from time import perf_counter


class SuffixIndex:
    """Index a fixed-length key only after observing its whole continuation."""

    def __init__(self, history, suffix, continuation, *, reject_backoff=True):
        self.history = []
        self.suffix = suffix
        self.continuation = continuation
        self.reject_backoff = reject_backoff
        self.next_start = 0
        self.starts = {}
        self.hit_rounds = []
        self.first_rejections = 0
        self.setup_s = 0.0
        self.extend(history)

    def seed(self, prompt, first):
        start = perf_counter()
        self.extend(prompt)
        self.extend([first])
        self.setup_s = perf_counter() - start

    def extend(self, verified):
        self.history.extend(verified)
        stop = len(self.history) - self.suffix - self.continuation + 1
        for start in range(self.next_start, stop):
            after = start + self.suffix
            key = tuple(self.history[start:after])
            previous = self.starts.setdefault(key, after)
            if previous is not None and previous != after:
                first = self.history[previous : previous + self.continuation]
                second = self.history[after : after + self.continuation]
                if first != second:
                    self.starts[key] = None
        self.next_start = max(self.next_start, stop)

    def lookup_start(self):
        return self.starts.get(tuple(self.history[-self.suffix :]))

    def proposals(self):
        start = self.lookup_start()
        if start is not None:
            return self.history[start : start + self.continuation]
        return None

    def wide_proposals(self, wide_index, prefix):
        """Return a longer match and whether it came from the ordinary index."""
        longer = wide_index.proposals()
        if longer is not None and longer[: len(prefix)] == prefix:
            return longer, True
        return None, False

    def observe(
        self, emitted, round_index, first_rejected, finished, *, reject_key=True
    ):
        if first_rejected is not None:
            self.hit_rounds.append(round_index)
            self.first_rejections += first_rejected
            if first_rejected and reject_key and self.reject_backoff:
                self.starts[tuple(self.history[-self.suffix :])] = None
        if not finished:
            self.extend(emitted)


def _runtime():
    if __package__ == "strategies":
        import drafter_q35
    else:
        from .. import drafter_q35

    return drafter_q35


def generate(
    model,
    block,
    drafter,
    prompt,
    max_tokens,
    size,
    eos,
    *,
    suffix=12,
    reject_backoff=True,
    index_factory=SuffixIndex,
    wide=False,
):
    """Use the normal packed verifier, substituting only the proposed tokens."""
    if suffix < 8 or size < 2:
        raise ValueError(
            "a copy proposal needs at least eight matching tokens and a draft slot"
        )
    if wide and size != 8:
        raise ValueError("wide lookup checks extend the eight-slot decoder")
    index = index_factory((), suffix, size - 1, reject_backoff=reject_backoff)
    long_index = (
        SuffixIndex((), suffix, 15, reject_backoff=reject_backoff) if wide else None
    )
    tokens, stats = _runtime()._generate_linear(
        model,
        block,
        drafter,
        prompt,
        max_tokens,
        size,
        eos,
        copy_index=index,
        wide_index=long_index,
    )
    kept = [stats["kept"][i] for i in index.hit_rounds]
    wide_rounds = set(long_index.hit_rounds) if long_index is not None else set()
    stats.update(
        ngram_rounds=len(index.hit_rounds),
        ngram_kept=sum(kept),
        ngram_first_rejections=index.first_rejections,
        ngram_full_accepts=sum(
            advance == (16 if i in wide_rounds else size)
            for i, advance in zip(index.hit_rounds, kept)
        ),
        ngram_round_indices=index.hit_rounds,
        ngram_round_ms=[stats["round_ms"][i] for i in index.hit_rounds],
        ngram_setup_s=index.setup_s
        + (long_index.setup_s if long_index is not None else 0),
    )
    if long_index is not None:
        stats.update(
            ngram_wide_rounds=len(wide_rounds), ngram_wide_indices=long_index.hit_rounds
        )
    return tokens, stats
