"""Use the installed NVIDIA driver to JIT PTX and run the packed kernel."""
import ctypes as ct
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


U32 = ct.c_uint32
U64 = ct.c_uint64
PTR = ct.c_void_p
I32 = ct.c_int
SIZE = ct.c_size_t
F32 = ct.c_float


class Cuda:
    def __init__(self, ordinal=0):
        self.lib = ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        self.device = I32()
        self.context = PTR()
        self.previous = PTR()
        self.module = PTR()
        self.kernel = PTR()
        self.weights = []
        self.resident_bytes = 0
        self.call("cuInit", U32(0))
        self.call("cuDeviceGet", ct.byref(self.device), I32(ordinal))
        self.call("cuCtxGetCurrent", ct.byref(self.previous))
        self.call("cuDevicePrimaryCtxRetain", ct.byref(self.context), self.device)
        try:
            self.call("cuCtxSetCurrent", self.context)
            ptx = ct.create_string_buffer(Path(__file__).with_name("matvec.ptx").read_bytes())
            self.call("cuModuleLoadData", ct.byref(self.module), ptx)
            self.call("cuModuleGetFunction", ct.byref(self.kernel), self.module, ct.c_char_p(b"packed_matvec"))
        except Exception:
            self.close()
            raise

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            error = ct.c_char_p()
            self.lib.cuGetErrorName(status, ct.byref(error))
            label = error.value.decode() if error.value else str(status)
            raise RuntimeError(f"{name} failed with {label}")

    def device_name(self):
        name = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", name, I32(len(name)), self.device)
        return name.value.decode()

    def pci_bus_id(self):
        bus_id = ct.create_string_buffer(32)
        self.call("cuDeviceGetPCIBusId", bus_id, I32(len(bus_id)), self.device)
        return bus_id.value.decode()

    def memory(self):
        free, total = SIZE(), SIZE()
        self.call("cuMemGetInfo_v2", ct.byref(free), ct.byref(total))
        return free.value, total.value

    def allocate(self, nbytes):
        pointer = U64()
        self.call("cuMemAlloc_v2", ct.byref(pointer), SIZE(nbytes))
        return pointer

    def write(self, pointer, array):
        self.call("cuMemcpyHtoD_v2", pointer, PTR(array.ctypes.data), SIZE(array.nbytes))

    def read(self, pointer, array):
        self.call("cuMemcpyDtoH_v2", PTR(array.ctypes.data), pointer, SIZE(array.nbytes))
        return array

    def copy(self, destination, source, nbytes):
        self.call("cuMemcpyDtoDAsync_v2", destination, source, SIZE(nbytes), PTR())

    def module_function(self, module, name):
        function = PTR()
        self.call("cuModuleGetFunction", ct.byref(function), module, ct.c_char_p(name.encode("ascii")))
        return function

    def load_module(self, source):
        module = PTR()
        buffer = ct.create_string_buffer(source.encode("ascii"))
        log = ct.create_string_buffer(8192)
        options = (I32 * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER and its size.
        values = (PTR * 2)(ct.addressof(log), len(log))
        try:
            self.call("cuModuleLoadDataEx", ct.byref(module), buffer, U32(2), options, values)
        except RuntimeError as error:
            raise RuntimeError(f"{error}\n{log.value.decode(errors='replace')}") from error
        return module

    def launch(self, function, grid, threads, *args, shared=0, stream=None):
        params = (PTR * len(args))(*(ct.addressof(arg) for arg in args))
        self.call("cuLaunchKernel", function, *(U32(i) for i in grid),
                  U32(threads), U32(1), U32(1), U32(shared),
                  stream if stream is not None else PTR(), params, PTR())

    def packed_project(self, slab, x, experts, out, slots, x_div, stream=None, *, batch=1, stride=0):
        self.launch(
            self.kernel, ((slab.rows + 3) // 4, slots, batch), 128,
            x, slab.codes, slab.scales, slab.table, experts, out,
            U32(slab.rows), U32(slab.width), U32(slab.words), U32(slab.bits),
            U32(slab.group), U32(slab.hash_shift), U32(slab.multiplier),
            U32(x_div), U32(slab.lloyd), U64(), U64(), U32(stride), stream=stream,
        )

    def routed_project(self, x, experts, out, slots, x_div, rows, width, descriptors, mapping, *, batch=1, stride=0):
        self.launch(
            self.kernel, ((rows + 3) // 4, slots, batch), 128,
            x, U64(), U64(), U64(), experts, out,
            U32(rows), U32(width), U32(), U32(), U32(), U32(), U32(), U32(x_div),
            U32(), descriptors, mapping, U32(stride),
        )

    def upload(self, array):
        array = np.ascontiguousarray(array)
        pointer = self.allocate(array.nbytes)
        self.weights.append(pointer)
        self.resident_bytes += array.nbytes
        self.write(pointer, array)
        return pointer

    @contextmanager
    def temporary(self, *sizes):
        pointers = []
        try:
            for size in sizes:
                pointers.append(self.allocate(size))
            yield pointers
        finally:
            for pointer in reversed(pointers):
                self.call("cuMemFree_v2", pointer)

    def project(self, inputs, slabs, routes, rows, x_div):
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        out = np.empty((routes[0].size, rows), dtype=np.float32)
        if out.shape[0] > 65535:
            raise ValueError("At most 65535 routed slots per CUDA call")
        with self.temporary(x.nbytes, routes[0].nbytes, out.nbytes) as (xp, ep, yp):
            self.write(xp, x)
            for slab, experts in zip(slabs, routes, strict=True):
                if not np.any(experts >= 0):
                    continue
                self.write(ep, experts)
                self.packed_project(slab, xp, ep, yp, out.shape[0], x_div)
            self.read(yp, out)
        return out

    def close(self):
        if self.context:
            self.call("cuCtxSetCurrent", self.context)
            for pointer in reversed(self.weights):
                self.call("cuMemFree_v2", pointer)
            self.weights.clear()
            self.resident_bytes = 0
            if self.module:
                self.call("cuModuleUnload", self.module)
            self.call("cuCtxSetCurrent", self.previous)
            self.call("cuDevicePrimaryCtxRelease_v2", self.device)
            self.context = PTR()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
