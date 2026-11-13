"""Continue a verified source span in an explicitly contiguous copy task."""

from . import ngram_draft


class SourceCursor(ngram_draft.SuffixIndex):
    """Try a previously verified source position on an ordinary lookup miss."""

    def __init__(self, history, suffix, continuation, *, reject_backoff=True):
        super().__init__(history, suffix, continuation, reject_backoff=reject_backoff)
        self.cursor = None
        self.proposed_from = None
        self.proposed_length = 0
        self.used_cursor = False
        self.cursor_rounds = []
        self.cursor_wide_rounds = []
        self.cursor_first_rejections = 0

    def _cursor_start(self, length):
        start = self.cursor
        if (
            start is not None
            and start + length <= len(self.history)
            and self.history[start - self.suffix : start]
            == self.history[-self.suffix :]
        ):
            return start
        return None

    def _choose(self, start, length, *, cursor):
        self.proposed_from = start
        self.proposed_length = length
        self.used_cursor = cursor
        return self.history[start : start + length]

    def proposals(self):
        self.proposed_from = None
        self.used_cursor = False
        start = self.lookup_start()
        if start is not None:
            return self._choose(start, self.continuation, cursor=False)
        start = self._cursor_start(self.continuation)
        if start is not None:
            return self._choose(start, self.continuation, cursor=True)
        return None

    def wide_proposals(self, wide_index, prefix):
        longer, ordinary = super().wide_proposals(wide_index, prefix)
        if ordinary:
            start = wide_index.lookup_start()
            return self._choose(start, len(longer), cursor=False), True

        length = wide_index.continuation
        start = self._cursor_start(length)
        if start is not None and self.history[start : start + len(prefix)] == prefix:
            return self._choose(start, length, cursor=True), False
        return None, False

    def observe(self, emitted, round_index, first_rejected, finished):
        if self.used_cursor:
            self.cursor_rounds.append(round_index)
            if self.proposed_length > self.continuation:
                self.cursor_wide_rounds.append(round_index)
            self.cursor_first_rejections += bool(first_rejected)

        # A full check emits the copied proposals plus the next anchor. Verify
        # the anchor also followed the source before advancing past it.
        self.cursor = None
        start = self.proposed_from
        if (
            not finished
            and start is not None
            and len(emitted) == self.proposed_length + 1
        ):
            end = start + len(emitted)
            if end <= len(self.history) and self.history[start:end] == list(emitted):
                self.cursor = end
        self.proposed_from = None
        self.used_cursor = False
        super().observe(emitted, round_index, first_rejected, finished)


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
    wide=False,
):
    """Opt in for a request that explicitly asks to copy one contiguous span."""
    index = SourceCursor((), suffix, size - 1, reject_backoff=reject_backoff)
    tokens, stats = ngram_draft.generate(
        model,
        block,
        drafter,
        prompt,
        max_tokens,
        size,
        eos,
        suffix=suffix,
        reject_backoff=reject_backoff,
        index_factory=lambda *_args, **_kwargs: index,
        wide=wide,
    )
    kept = [stats["kept"][i] for i in index.cursor_rounds]
    wide_rounds = set(index.cursor_wide_rounds)
    stats.update(
        cursor_rounds=len(kept),
        cursor_kept=sum(kept),
        cursor_full_accepts=sum(
            advance == (16 if i in wide_rounds else size)
            for i, advance in zip(index.cursor_rounds, kept)
        ),
        cursor_first_rejections=index.cursor_first_rejections,
        cursor_round_indices=index.cursor_rounds,
    )
    if wide:
        stats.update(
            cursor_wide_rounds=len(wide_rounds),
            cursor_wide_indices=index.cursor_wide_rounds,
        )
    return tokens, stats
