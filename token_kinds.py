"""What kind of token each position of a text predicts, for token weights: 0 a digit, 1 a digit inside a number that a
calculation's result states, 2 a symbol of arithmetic, 3 anything else."""

import torch


SYMBOLS = set("+-*/×÷^()%<>")


def kinds_of(ids, tok):
    """The kind of the token predicted at each position (positions 0 to T − 2 predict tokens 1 to T − 1)."""
    text = {i: tok.decode([i]).strip() for i in ids.unique().tolist()}
    out = torch.full(ids.shape, 3, dtype=torch.long)
    for w, row in enumerate(ids.tolist()):
        in_result = False
        for t, i in enumerate(row):
            s = text[i]
            if s.endswith("=") or s.endswith("\\="):
                in_result, out[w, t] = True, 2
            elif s.isdigit():
                out[w, t] = 0 if in_result else 1
            elif s in (",", ".", "", "$", "\\,"):
                pass  # a number or its formatting goes on
            else:
                in_result = False
                out[w, t] = 2 if s in SYMBOLS else 3
    return out[:, 1:]
