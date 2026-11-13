"""Change only the 16-slot packed gate/up launch geometry."""

from contextlib import contextmanager

if __package__ == "strategies":
    import block_q35 as B
else:
    from .. import block_q35 as B


class _WideProduct(B.BlockProduct):
    def __init__(self, original, shape):
        super().__init__(original.p, threadgroups=original.threadgroups)
        self.original = original
        self.wide_shape = shape

    def shape(self, tb):
        return self.wide_shape if tb == 2 else self.original.shape(tb)

    def __call__(self, x, tb=1):
        if tb == 2:
            return super().__call__(x, tb)
        return self.original(x, tb)


def stages(block):
    return [rec["m_in"] for rec in block.recs]


class WideProductGeometry:
    """Keep warmed 16-slot launch objects for a set of packed stages."""

    def __init__(self, products, shape):
        rb, sg, cs = shape = tuple(shape)
        self.changes = []
        for stage in products:
            old = stage.product
            if (
                rb < 1
                or not 1 <= sg <= 32
                or cs < 1
                or sg % cs
                or old.n // 32 % cs
                or (cs > 1 and sg * rb * 2 * 64 * 4 > 32768)
            ):
                raise ValueError(f"invalid packed product launch geometry: {shape}")
            self.changes.append((stage, old, _WideProduct(old, shape)))

    @contextmanager
    def installed(self):
        if any(stage.product is not old for stage, old, _ in self.changes):
            raise ValueError("packed products have another active experiment")
        try:
            for stage, _, new in self.changes:
                stage.product = new
            yield
        finally:
            for stage, old, _ in self.changes:
                stage.product = old


class GateUpGeometry(WideProductGeometry):
    def __init__(self, block, shape):
        super().__init__(stages(block), shape)
