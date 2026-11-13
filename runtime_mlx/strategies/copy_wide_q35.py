"""Use tuned 16-slot launches for the packed MLP products."""

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


class CopyWideGeometry:
    """Keep warmed 16-slot launches while preserving the down product's reduction order."""

    def __init__(self, block):
        self.changes = []
        for rec in block.recs:
            gate_up, down = rec["m_in"], rec["m_down"]
            if down.product.shape(2)[2] != 8 or down.product.n // 32 % 8:
                raise ValueError("the 16-slot down launch requires eight column parts")
            for stage, shape in ((gate_up, (2, 16, 1)), (down, (2, 32, 8))):
                old = stage.product
                self.changes.append((stage, old, _WideProduct(old, shape)))

    @contextmanager
    def installed(self):
        if any(stage.product is not old for stage, old, _ in self.changes):
            raise ValueError("packed products already have an active launch override")
        try:
            for stage, _, new in self.changes:
                stage.product = new
            yield
        finally:
            for stage, old, _ in self.changes:
                stage.product = old
