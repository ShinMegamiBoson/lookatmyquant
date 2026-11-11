"""A math benchmark with exact answers: problems generated from a seed, each with one integer or fraction as its answer, so
scoring needs no judge and two models can be compared problem by problem.

Twelve families at three difficulties. Every answer is computed exactly, in Python integers and fractions, and each family
has a second, independent check in check().

    python math_bench.py                  # the checks, and a few sample problems
    problems = make_problems(seed=0, per_cell=15)
    score(problems[0], model_output_text)
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}, as an integer or a fraction in lowest terms."


def signed(rng, lo, hi):
    return rng.randint(lo, hi) * rng.choice((-1, 1))


def power_mod_naive(a, b, m):
    r = 1 % m
    for _ in range(b):
        r = r * a % m
    return r


def det(M):
    """Exact determinant by cofactors."""
    if len(M) == 1:
        return Fraction(M[0][0])
    return sum((-1) ** j * M[0][j] * det([r[:j] + r[j + 1 :] for r in M[1:]]) for j in range(len(M)))


def fmt_matrix(M):
    return "[" + ", ".join("[" + ", ".join(str(x) for x in r) + "]" for r in M) + "]"


def add_sub(rng, d):
    n = (6, 9, 12)[d]
    a, b, c = [rng.randint(10 ** (n - 1), 10**n - 1) for _ in range(3)]
    return f"Compute {a} + {b} - {c}.", a + b - c, {"terms": (a, b, -c)}


def multiply(rng, d):
    n = (3, 4, 5)[d]
    a, b = [rng.randint(10 ** (n - 1), 10**n - 1) for _ in range(2)]
    return f"Compute {a} × {b}.", a * b, {"a": a, "b": b}


def expression(rng, d):
    ops = (5, 8, 12)[d]

    def build(n):
        if n == 0:
            v = rng.randint(1, 19)
            return str(v), v
        k = rng.randint(0, n - 1)
        ls, lv = build(k)
        rs, rv = build(n - 1 - k)
        op = rng.choice("+-*")
        return f"({ls} {op} {rs})", {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]

    s, v = build(ops)
    return f"Evaluate {s[1:-1]}.", v, {"text": s}


def modpow(rng, d):
    a, b, m = rng.randint(2, (20, 200, 2000)[d]), rng.randint((5, 20, 60)[d], (15, 60, 200)[d]), rng.randint((7, 50, 300)[d], (50, 500, 3000)[d])
    return f"Compute {a}^{b} modulo {m}, as an integer from 0 to {m - 1}.", pow(a, b, m), {"a": a, "b": b, "m": m}


def gcd(rng, d):
    n = (3, 5, 7)[d]
    g = rng.randint(2, 10**n)
    a, b = g * rng.randint(2, 10 ** (n // 2 + 1)), g * rng.randint(2, 10 ** (n // 2 + 1))
    return f"Find the greatest common divisor of {a} and {b}.", math.gcd(a, b), {"a": a, "b": b}


def linear_system(rng, d):
    n = (2, 3, 4)[d]
    while True:
        A = [[signed(rng, 0, 9) for _ in range(n)] for _ in range(n)]
        if det(A) != 0:
            break
    x = [signed(rng, 0, 12) for _ in range(n)]
    b = [sum(a * v for a, v in zip(r, x)) for r in A]
    names = "xyzw"[:n]
    eqs = "; ".join(" + ".join(f"({c}){names[j]}" for j, c in enumerate(r)) + f" = {v}" for r, v in zip(A, b))
    return f"Solve this system and give the value of {names[0]}: {eqs}.", x[0], {"A": A, "b": b, "x": x}


def determinant(rng, d):
    n, hi = ((2, 99), (3, 15), (4, 7))[d]
    M = [[signed(rng, 0, hi) for _ in range(n)] for _ in range(n)]
    return f"Compute the determinant of the matrix {fmt_matrix(M)}.", int(det(M)), {"M": M}


def polynomial(rng, d):
    deg = (3, 5, 7)[d]
    c = [signed(rng, 0, 9) for _ in range(deg)] + [rng.randint(1, 9)]
    x = signed(rng, 1, (5, 6, 7)[d])
    terms = " + ".join(f"({a})x^{i}" if i else f"({a})" for i, a in enumerate(c))
    return f"Evaluate p(x) = {terms} at x = {x}.", sum(a * x**i for i, a in enumerate(c)), {"c": c, "x": x}


def binomial(rng, d):
    n = rng.randint((10, 20, 30)[d], (18, 30, 45)[d])
    k = rng.randint(3, n // 2)
    return f"How many ways are there to choose {k} items from {n} distinct items?", math.comb(n, k), {"n": n, "k": k}


def base_convert(rng, d):
    b = rng.choice((2, 3, 5, 7, 8, 9))
    n = (8, 12, 18)[d]
    digits = [rng.randint(1, b - 1)] + [rng.randint(0, b - 1) for _ in range(n - 1)]
    s = "".join(map(str, digits))
    return f"Convert the base-{b} number {s} to base 10.", int(s, b), {"digits": digits, "b": b}


def fractions(rng, d):
    n = (3, 4, 6)[d]
    t = [Fraction(signed(rng, 1, 30), rng.randint(2, (12, 25, 40)[d])) for _ in range(n)]
    s = " + ".join(f"({f.numerator}/{f.denominator})" for f in t)
    return f"Compute {s}.", sum(t), {"terms": [(f.numerator, f.denominator) for f in t]}


def recurrence(rng, d):
    p, q, r = rng.randint(1, 4), signed(rng, 1, 3), signed(rng, 0, 6)
    a0, a1 = signed(rng, 0, 9), signed(rng, 0, 9)
    n = rng.randint((6, 10, 15)[d], (9, 14, 20)[d])
    a = [a0, a1]
    for _ in range(n - 1):
        a.append(p * a[-1] + q * a[-2] + r)
    return f"A sequence has a_0 = {a0}, a_1 = {a1}, and a_k = ({p})a_(k-1) + ({q})a_(k-2) + ({r}) for k ≥ 2. Find a_{n}.", a[n], {"p": p, "q": q, "r": r, "a0": a0, "a1": a1, "n": n}


FAMILIES = {f.__name__: f for f in (add_sub, multiply, expression, modpow, gcd, linear_system, determinant, polynomial, binomial, base_convert, fractions, recurrence)}


def make_problems(seed=0, per_cell=15):
    """The same seed gives the same problems on any machine: each one has its own generator, seeded by a string. Small
    families can repeat a problem, so a repeat draws again."""
    out, seen = [], set()
    for name, f in FAMILIES.items():
        for d in range(3):
            for i in range(per_cell):
                draw = 0
                while True:
                    q, a, data = f(random.Random(f"{seed}/{name}/{d}/{i}/{draw}"), d)
                    if q not in seen:
                        break
                    draw += 1
                seen.add(q)
                a = Fraction(a)
                out.append({"id": f"{name}/{d}/{i}", "family": name, "difficulty": d, "question": q, "answer": f"{a.numerator}/{a.denominator}" if a.denominator != 1 else str(a.numerator), "data": data})
    return out


def parse(text):
    """The last \\boxed{...} after any thinking, as an exact number, or None."""
    text = text.rsplit("</think>", 1)[-1]
    hits, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i < 0:
            break
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            hits.append(text[i + 7 : j - 1])
        i = j
    if not hits:
        return None
    s = hits[-1].strip().replace(",", "").replace(" ", "").replace("−", "-").replace("\\,", "").replace("$", "")
    m = re.fullmatch(r"(-?)\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        s = f"{m.group(1)}{m.group(2)}/{m.group(3)}"
    m = re.fullmatch(r"(-?\d+)(?:/(-?\d+))?", s)
    if not m or m.group(2) in ("0", "-0"):
        return None
    return Fraction(int(m.group(1)), int(m.group(2) or 1))


def score(problem, text):
    """(correct, the parsed answer or None). A fraction must be equal in value: lowest terms is asked for, not required."""
    got = parse(text)
    return got is not None and got == Fraction(problem["answer"]), None if got is None else str(got)


def check(p):
    """A second computation of the answer, different from the one that made it."""
    k, d, a = p["family"], p["data"], Fraction(p["answer"])
    if k == "add_sub":
        return sum(d["terms"]) == a
    if k == "multiply":
        return sum(int(c) * 10**i * d["a"] for i, c in enumerate(reversed(str(d["b"])))) == a
    if k == "expression":
        return eval(d["text"]) == a  # our own generated text: digits, parentheses, + - *
    if k == "modpow":
        return power_mod_naive(d["a"], d["b"], d["m"]) == a
    if k == "gcd":
        x, y = d["a"], d["b"]
        while y:
            x, y = y, x - y * (x // y)
        return x == a
    if k == "linear_system":
        A, b, x = d["A"], d["b"], d["x"]
        n = len(A)
        if not all(sum(A[i][j] * x[j] for j in range(n)) == b[i] for i in range(n)):
            return False
        return Fraction(det([[b[i] if j == 0 else A[i][j] for j in range(n)] for i in range(n)])) / det(A) == a  # Cramer's rule
    if k == "determinant":  # exact Gaussian elimination
        M, sign, out = [[Fraction(x) for x in r] for r in d["M"]], 1, Fraction(1)
        n = len(M)
        for c in range(n):
            piv = next((r for r in range(c, n) if M[r][c] != 0), None)
            if piv is None:
                return a == 0
            if piv != c:
                M[c], M[piv], sign = M[piv], M[c], -sign
            out *= M[c][c]
            for r in range(c + 1, n):
                f = M[r][c] / M[c][c]
                M[r] = [u - f * v for u, v in zip(M[r], M[c])]
        return sign * out == a
    if k == "polynomial":  # Horner
        v = 0
        for c in reversed(d["c"]):
            v = v * d["x"] + c
        return v == a
    if k == "binomial":  # Pascal's rule
        row = [1]
        for _ in range(d["n"]):
            row = [1] + [u + v for u, v in zip(row, row[1:])] + [1]
        return row[d["k"]] == a
    if k == "base_convert":
        return sum(x * d["b"] ** i for i, x in enumerate(reversed(d["digits"]))) == a
    if k == "fractions":
        num, den = 0, 1
        for n_, m in d["terms"]:
            num, den = num * m + n_ * den, den * m
        return Fraction(num, den) == a
    if k == "recurrence":
        x, y = d["a0"], d["a1"]
        for _ in range(d["n"]):
            x, y = y, d["p"] * y + d["q"] * x + d["r"]
        return x == a
    raise KeyError(k)


if __name__ == "__main__":
    P = make_problems()
    bad = [p["id"] for p in P if not check(p)]
    assert not bad, bad[:5]
    assert [p["answer"] for p in make_problems()] == [p["answer"] for p in P], "the same seed must give the same problems"
    assert len({p["question"] for p in P}) == len(P), "duplicate problems"
    for text, want in (("<think>2</think> so \\boxed{1,234}", "1234"), ("\\boxed{-3/4}", "-3/4"), ("\\boxed{\\frac{3}{4}}", "3/4"), ("\\boxed{-\\frac{3}{4}}", "-3/4"), ("no answer", None), ("<think>\\boxed{5}</think>then none", None), ("\\boxed{2} then \\boxed{7}", "7")):
        got = parse(text)
        assert (None if got is None else str(got)) == want, (text, got)
    print(len(P), "problems, every answer checked a second way")
    for name in FAMILIES:
        p = next(x for x in P if x["family"] == name and x["difficulty"] == 2)
        print(f"  {name}: {p['question'][:110]} -> {p['answer'][:40]}")
