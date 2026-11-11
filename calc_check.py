"""A checker for the arithmetic a worked solution states: finds each "a op b = c" chain in a text and its first wrong result."""

from __future__ import annotations

import re
from fractions import Fraction


NUM = r"-?\d[\d,]*(?:\.\d+)?"


OPS = {"+": "+", "-": "-", "−": "-", "×": "*", "*": "*", "x": "*", "·": "*", "\\times": "*", "\\cdot": "*", "/": "/", "÷": "/", "\\div": "/"}


OP = r"(?:\\times|\\cdot|\\div|[+\-−×*x·/÷])"


TIMES = r"(?:\\times|\\cdot|[×*x·])"


# a result followed by an operator or a bracket is an expansion ("= 1568 × (7000 + 574)"), not a value
CHAIN = re.compile(rf"(?<![\w.^])({NUM}(?:\s*{OP}\s*{NUM})+)\s*=\s*({NUM})(?![\d.]*[ \t]*(?:{OP}|\(|\d))")


DIVISION = re.compile(rf"(?<![\w.^])({NUM})\s*=\s*({NUM})\s*{TIMES}\s*({NUM})\s*\+\s*({NUM})(?![\d.]*[ \t]*(?:{OP}|\(|\d))")


MOD = re.compile(rf"(?<![\w.^])({NUM})\s*(?:\\bmod|\\pmod|mod|\\%|%)\s*\{{?({NUM})\}}?\s*=\s*({NUM})(?![\d.])")


def number(s):
    return Fraction(s.replace(",", ""))


def value(expr):
    """The chain's value, × and / before + and −, exact; None if it is not plain arithmetic."""
    s, toks = expr.replace(" ", ""), []
    while s:  # number and operator in turn; only the first number may carry a sign
        m = re.match((NUM if not toks else NUM[2:]) if len(toks) % 2 == 0 else OP, s)
        if not m:
            return None
        toks.append(m.group(0))
        s = s[m.end():]
    if len(toks) < 3 or len(toks) % 2 == 0:
        return None
    try:
        terms, ops = [number(toks[0])], []
        for op, t in zip(toks[1::2], toks[2::2]):
            o = OPS.get(op)
            if o is None:
                return None
            if o in "*/":
                terms[-1] = terms[-1] * number(t) if o == "*" else terms[-1] / number(t)
            else:
                terms.append(number(t))
                ops.append(o)
        out = terms[0]
        for o, t in zip(ops, terms[1:]):
            out = out + t if o == "+" else out - t
        return out
    except (ZeroDivisionError, ValueError):
        return None


def first_wrong(text):
    """(position where the corrected text resumes, the exact result to write there) for the first wrong stated result, or None."""
    found = []
    for m in CHAIN.finditer(text):
        v = value(m.group(1))
        if v is None or v.denominator != 1 or m.end(2) >= len(text) or re.match(r",\d|\s*\^", text[m.end(2) :]) or re.search(r"\d,$|[A-Za-z)\]}]\s*[+\-−]\s*$", text[: m.start(1)]):
            continue  # (numbers with thousands commas cut short, "350,800 + 551 = 351,351"; a power next, "= 2^2 × …"; the tail of an algebraic sum, "-2y + 8 - 81")
        w = number(m.group(2))
        if w == v or v < 0 and w.denominator == 1 and w == v % 10 ** len(m.group(2).strip("-")):
            continue  # (a column subtraction's borrow: "0 − 8 = 2")
        found.append((m.start(2), str(int(v))))
    for m in DIVISION.finditer(text):
        a, x, y, r = (number(g) for g in m.groups())
        if 0 <= a - x * y < max(x, y) and a - x * y != r and m.end(4) < len(text):
            found.append((m.start(4), str(int(a - x * y))))
    for m in MOD.finditer(text):
        a, b, r = (number(g) for g in m.groups())
        if b and a.denominator == b.denominator == 1 and a % b != r and m.end(3) < len(text):
            found.append((m.start(3), str(int(a % b))))
    return min(found) if found else None
