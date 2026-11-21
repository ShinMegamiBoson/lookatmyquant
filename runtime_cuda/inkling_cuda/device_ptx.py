"""CUDA text decoder primitives, assembled by the NVIDIA driver from PTX."""

HEADER = (".version 7.8\n.target sm_80\n.address_size 64\n"
          ".extern .shared .align 4 .b8 dynamic_shared[];\n")


def entry(name, args, body, declarations="", batched=""):
    params, loads = [], []
    for ty, declarations_for_type in args.items():
        for index, label in enumerate(declarations_for_type.split(), 1):
            tag = {"u64": "d", "u32": "n", "f32": "a"}[ty]
            params.append(f".param .{ty} {label}")
            loads.append(f"ld.param.{ty} %{tag}{index}, [{label}];")
    if batched:
        loads.extend(("mov.u32 %r59, %ctaid.z;", "mul.wide.u32 %rd29, %r59, %batch_stride;"))
        for label in batched.split():
            reg = args["u64"].split().index(label) + 1
            loads.append(f"add.u64 %d{reg}, %d{reg}, %rd29;")
    return (
        f".visible .entry {name}({', '.join(params)}, .param .u32 row_stride) {{\n"
        ".reg .pred %p<12>; .reg .b16 %half; .reg .b32 %r<64>, %n<24>, %batch_stride;\n"
        ".reg .b64 %rd<32>, %d<40>; .reg .f32 %f<48>, %a<20>;\n"
        + declarations + "\nld.param.u32 %batch_stride, [row_stride];\n"
        + "\n".join(loads) + "\n" + body + "\n}\n"
    )


def warp_reduce(value, op):
    return "\n".join(f"""
    shfl.sync.bfly.b32 %f47, {value}, {i}, 31, -1;
    {op}.f32 {value}, {value}, %f47;""" for i in (16, 8, 4, 2, 1))


def block_reduce(value, op, label):
    """Reduce four warps, then broadcast to all lanes of a 128-thread block."""
    neutral = "0f00000000" if op == "add" else "0fFF800000"
    return warp_reduce(value, op) + f"""
    mov.u32 %r60, %tid.x;
    mov.u32 %r61, %laneid;
    shr.u32 %r62, %r60, 5;
    mov.u64 %rd30, reduction;
    mad.wide.u32 %rd31, %r62, 4, %rd30;
    setp.eq.u32 %p10, %r61, 0;
    @%p10 st.shared.f32 [%rd31], {value};
    bar.sync 0;
    setp.ge.u32 %p11, %r60, 32;
    @%p11 bra {label}_join;
    mov.b32 {value}, {neutral};
    setp.lt.u32 %p11, %r60, 4;
    mad.wide.u32 %rd31, %r60, 4, %rd30;
    @%p11 ld.shared.f32 {value}, [%rd31];
    {warp_reduce(value, op)}
    @%p10 st.shared.f32 [%rd30], {value};
{label}_join:
    bar.sync 0;
    ld.shared.f32 {value}, [%rd30];
    bar.sync 0;
"""


REDUCTION = ".shared .align 4 .b32 reduction[4];"


def vector(name, pointers, floats, body):
    return entry(name, {"u64": pointers, "u32": "length", "f32": floats}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mul.wide.u32 %rd1, %r1, 4;
""" + body + "\n    ret;", batched=" ".join(p for p in pointers.split() if p != "order"))


def sigmoid(src, dest):
    return f"""
    mul.f32 {dest}, {src}, 0fBFB8AA3B;
    ex2.approx.f32 {dest}, {dest};
    add.f32 {dest}, {dest}, 0f3F800000;
    rcp.approx.f32 {dest}, {dest};
"""


def bf16(src, dest):
    return f"cvt.rn.bf16.f32 %half, {src}; cvt.f32.bf16 {dest}, %half;"


ADD = vector("vec_add", "x y out", "", """
    add.u64 %rd2, %d1, %rd1;
    add.u64 %rd3, %d2, %rd1;
    add.u64 %rd4, %d3, %rd1;
    ld.global.f32 %f1, [%rd2];
    ld.global.f32 %f2, [%rd3];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd4], %f1;
""")

CAST = vector("vec_bf16", "x out", "", f"""
    add.u64 %rd2, %d1, %rd1;
    add.u64 %rd3, %d2, %rd1;
    ld.global.f32 %f1, [%rd2];
    {bf16('%f1', '%f1')}
    st.global.f32 [%rd3], %f1;
""")

ATTN_GATE = vector("attention_gate", "x gate out", "", f"""
    add.u64 %rd2, %d1, %rd1;
    add.u64 %rd3, %d2, %rd1;
    add.u64 %rd4, %d3, %rd1;
    ld.global.f32 %f1, [%rd2];
    ld.global.f32 %f2, [%rd3];
    {sigmoid('%f2', '%f3')}
    mul.f32 %f1, %f1, %f3;
    st.global.f32 [%rd4], %f1;
""")

MESSAGE = vector("message_mix", "x message forget addend out", "", f"""
    add.u64 %rd2, %d1, %rd1;
    add.u64 %rd3, %d2, %rd1;
    add.u64 %rd4, %d3, %rd1;
    add.u64 %rd5, %d4, %rd1;
    add.u64 %rd6, %d5, %rd1;
    ld.global.f32 %f1, [%rd2];
    ld.global.f32 %f2, [%rd3];
    ld.global.f32 %f3, [%rd4];
    ld.global.f32 %f4, [%rd5];
    {bf16('%f3', '%f3')}
    {bf16('%f4', '%f4')}
    tanh.approx.f32 %f3, %f3;
    tanh.approx.f32 %f4, %f4;
    sub.f32 %f3, 0f3F800000, %f3;
    mul.f32 %f1, %f1, %f3;
    fma.rn.f32 %f1, %f2, %f4, %f1;
    st.global.f32 [%rd6], %f1;
""")


def normalize(name, clip=False):
    finish = """
    sqrt.approx.f32 %f3, %f1;
    add.f32 %f3, %f3, 0f358637BD;
    div.rn.f32 %f3, %a1, %f3;
    min.f32 %f3, %f3, 0f3F800000;
""" if clip else """
    cvt.rn.f32.u32 %f3, %n1;
    div.rn.f32 %f3, %f1, %f3;
    add.f32 %f3, %f3, %a1;
    rsqrt.approx.f32 %f3, %f3;
"""
    read = """
    mad.wide.u32 %rd3, %r1, 4, %rd1;
    ld.global.f32 %f2, [%rd3];
""" + ("""
    @%p2 bra no_scale_{label};
    mad.wide.u32 %rd4, %r4, 4, %d2;
    ld.global.f32 %f4, [%rd4];
    mul.f32 %f2, %f2, %f4;
no_scale_{label}:
""" if clip else "")
    weight = "" if clip else """
    @%p2 bra no_weight;
    mad.wide.u32 %rd4, %r4, 4, %d2;
    ld.global.f32 %f4, [%rd4];
    mul.f32 %f2, %f2, %f4;
no_weight:
"""
    return entry(name, {"u64": "x weight out", "u32": "width", "f32": "epsilon_or_threshold factor"}, f"""
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mul.lo.u32 %r3, %r2, %n1;
    mad.wide.u32 %rd1, %r3, 4, %d1;
    mad.wide.u32 %rd2, %r3, 4, %d3;
    setp.eq.u64 %p2, %d2, 0;
    mov.b32 %f1, 0;
sum_loop:
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 bra sum_finish;
    add.u32 %r4, %r1, %r3;
    {read.format(label="sum")}
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u32 %r1, %r1, 128;
    bra sum_loop;
sum_finish:
    {block_reduce('%f1', 'add', 'norm')}
    {finish}
    mul.f32 %f3, %f3, %a2;
    mov.u32 %r1, %tid.x;
write_loop:
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    add.u32 %r4, %r1, %r3;
    {read.format(label="write")}
    {weight}
    mul.f32 %f2, %f2, %f3;
    mad.wide.u32 %rd3, %r1, 4, %rd2;
    st.global.f32 [%rd3], %f2;
    add.u32 %r1, %r1, 128;
    bra write_loop;
""", REDUCTION, batched="x out")


def dense(name, split_bf16=False):
    initialize = """
    shr.u32 %r6, %n2, 1;
    mad.wide.u32 %rd4, %r6, 4, %rd1;
    mad.wide.u32 %rd5, %r6, 4, %d1;
    mov.b32 %f4, 0;
""" if split_bf16 else "mov.u32 %r6, %n2;"
    second_dot = """
    mad.wide.u32 %rd2, %r5, 4, %rd4;
    mad.wide.u32 %rd3, %r5, 4, %rd5;
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    fma.rn.f32 %f4, %f2, %f3, %f4;
""" if split_bf16 else ""
    merge = f"""
    {warp_reduce('%f4', 'add')}
    {bf16('%f1', '%f1')}
    {bf16('%f4', '%f4')}
    add.f32 %f1, %f1, %f4;
    {bf16('%f1', '%f1')}
""" if split_bf16 else ""
    return entry(name, {"u64": "x matrix out", "u32": "rows width"}, f"""
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    shr.u32 %r3, %r2, 5;
    mad.lo.u32 %r3, %r1, 4, %r3;
    setp.ge.u32 %p1, %r3, %n1;
    @%p1 ret;
    mul.lo.u32 %r4, %r3, %n2;
    mad.wide.u32 %rd1, %r4, 4, %d2;
    mov.u32 %r5, %laneid;
    mov.b32 %f1, 0;
    {initialize}
loop:
    setp.ge.u32 %p2, %r5, %r6;
    @%p2 bra finish;
    mad.wide.u32 %rd2, %r5, 4, %rd1;
    mad.wide.u32 %rd3, %r5, 4, %d1;
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    {second_dot}
    add.u32 %r5, %r5, 32;
    bra loop;
finish:
    {warp_reduce('%f1', 'add')}
    {merge}
    mov.u32 %r5, %laneid;
    setp.ne.u32 %p3, %r5, 0;
    @%p3 ret;
    mad.wide.u32 %rd2, %r3, 4, %d3;
    st.global.f32 [%rd2], %f1;
    ret;
""", batched="x out")


ROTATE = entry("hadamard", {"u64": "x signs paley out", "u32": "width base inverse", "f32": "scale"}, """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mul.lo.u32 %r3, %r2, %n1;
    mad.wide.u32 %rd1, %r3, 4, %d1;
    mad.wide.u32 %rd2, %r3, 4, %d4;
    mov.u64 %rd3, dynamic_shared;
    setp.ne.u32 %p4, %n3, 0;
    setp.eq.u32 %p5, %n2, 1;
load_loop:
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 bra loaded;
    @%p5 bra scalar_input;
    rem.u32 %r4, %r1, %n2;
    sub.u32 %r5, %r1, %r4;
    mov.u32 %r6, 0;
    mov.b32 %f1, 0;
paley_loop:
    add.u32 %r7, %r5, %r6;
    mad.wide.u32 %rd4, %r7, 4, %rd1;
    ld.global.f32 %f2, [%rd4];
    @%p4 bra paley_matrix;
    mad.wide.u32 %rd4, %r7, 4, %d2;
    ld.global.f32 %f3, [%rd4];
    mul.f32 %f2, %f2, %f3;
paley_matrix:
    mad.lo.u32 %r8, %r4, %n2, %r6;
    mad.lo.u32 %r9, %r6, %n2, %r4;
    selp.u32 %r8, %r9, %r8, %p4;
    mad.wide.u32 %rd4, %r8, 4, %d3;
    ld.global.f32 %f3, [%rd4];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p2, %r6, %n2;
    @%p2 bra paley_loop;
    bra store;
scalar_input:
    mad.wide.u32 %rd4, %r1, 4, %rd1;
    ld.global.f32 %f1, [%rd4];
    @%p4 bra store;
    mad.wide.u32 %rd4, %r1, 4, %d2;
    ld.global.f32 %f3, [%rd4];
    mul.f32 %f1, %f1, %f3;
store:
    mad.wide.u32 %rd4, %r1, 4, %rd3;
    st.shared.f32 [%rd4], %f1;
    add.u32 %r1, %r1, 128;
    bra load_loop;
loaded:
    bar.sync 0;
    mov.u32 %r8, 1;
    mul.lo.u32 %r9, %r8, %n2;
butterfly:
    shl.b32 %r10, %r9, 1;
    setp.gt.u32 %p1, %r10, %n1;
    @%p1 bra done;
    mov.u32 %r1, %tid.x;
    shr.u32 %r11, %n1, 1;
butterfly_loop:
    setp.ge.u32 %p1, %r1, %r11;
    @%p1 bra butterfly_sync;
    div.u32 %r12, %r1, %r9;
    mad.lo.u32 %r13, %r12, %r9, %r1;
    add.u32 %r14, %r13, %r9;
    mad.wide.u32 %rd4, %r13, 4, %rd3;
    mad.wide.u32 %rd5, %r14, 4, %rd3;
    ld.shared.f32 %f1, [%rd4];
    ld.shared.f32 %f2, [%rd5];
    add.f32 %f3, %f1, %f2;
    sub.f32 %f4, %f1, %f2;
    st.shared.f32 [%rd4], %f3;
    st.shared.f32 [%rd5], %f4;
    add.u32 %r1, %r1, 128;
    bra butterfly_loop;
butterfly_sync:
    bar.sync 0;
    mov.u32 %r9, %r10;
    bra butterfly;
done:
    mov.u32 %r1, %tid.x;
write_loop:
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mad.wide.u32 %rd4, %r1, 4, %rd3;
    ld.shared.f32 %f1, [%rd4];
    mul.f32 %f1, %f1, %a1;
    @!%p4 bra write;
    mad.wide.u32 %rd4, %r1, 4, %d2;
    ld.global.f32 %f3, [%rd4];
    mul.f32 %f1, %f1, %f3;
write:
    mad.wide.u32 %rd4, %r1, 4, %rd2;
    st.global.f32 [%rd4], %f1;
    add.u32 %r1, %r1, 128;
    bra write_loop;
""", batched="x out")


PACKED_ROW = entry("packed_row", {
    "u64": "codes scales table out",
    "u32": "row width words bits group hash_shift multiplier lloyd",
}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n2;
    @%p1 ret;
    mul.lo.u32 %r3, %n1, %n3;
    mad.wide.u32 %rd1, %r3, 4, %d1;
    sub.u32 %r4, %n5, 1;
    and.b32 %r4, %r1, %r4;
    sub.u32 %r5, %r1, %r4;
    add.u32 %r5, %r5, %n5;
    sub.u32 %r5, %n2, %r5;
    setp.ne.u32 %p2, %n8, 0;
    selp.u32 %r5, %r1, %r5, %p2;
    mul.lo.u32 %r5, %r5, %n4;
    shr.u32 %r6, %r5, 5;
    and.b32 %r7, %r5, 31;
    mad.wide.u32 %rd2, %r6, 4, %rd1;
    ld.global.u32 %r8, [%rd2];
    add.u32 %r6, %r6, 1;
    mov.u32 %r9, 0;
    setp.lt.u32 %p3, %r6, %n3;
    @%p3 ld.global.u32 %r9, [%rd2+4];
    shf.r.clamp.b32 %r10, %r8, %r9, %r7;
    mov.u32 %r11, 1;
    shl.b32 %r11, %r11, %n4;
    sub.u32 %r11, %r11, 1;
    selp.u32 %r11, %r11, 4095, %p2;
    and.b32 %r10, %r10, %r11;
    setp.ne.u32 %p4, %n7, 0;
    @%p4 mul.lo.u32 %r10, %r10, %n7;
    @%p4 shr.u32 %r10, %r10, %n6;
    mad.lo.u32 %r10, %r10, %n5, %r4;
    mad.wide.u32 %rd3, %r10, 4, %d3;
    ld.global.f32 %f1, [%rd3];
    mad.wide.u32 %rd4, %n1, 2, %d2;
    ld.global.b16 %half, [%rd4];
    cvt.f32.f16 %f2, %half;
    mul.f32 %f1, %f1, %f2;
    mad.wide.u32 %rd5, %r1, 4, %d4;
    st.global.f32 [%rd5], %f1;
    ret;
""")


BLOCK_SCORES = entry("block_scores", {
    "u64": "states queries scores", "u32": "width count reads", "f32": "scale",
}, f"""
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    shr.u32 %r3, %r2, 5;
    mad.lo.u32 %r3, %r1, 4, %r3;
    mul.lo.u32 %r4, %n2, %n3;
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 ret;
    div.u32 %r4, %r3, %n2;
    rem.u32 %r5, %r3, %n2;
    mul.lo.u32 %r4, %r4, %n1;
    mul.lo.u32 %r5, %r5, %n1;
    mad.wide.u32 %rd1, %r4, 4, %d2;
    mad.wide.u32 %rd2, %r5, 4, %d1;
    mov.u32 %r6, %laneid;
    mov.b32 %f1, 0;
loop:
    setp.ge.u32 %p2, %r6, %n1;
    @%p2 bra finish;
    mad.wide.u32 %rd3, %r6, 4, %rd1;
    mad.wide.u32 %rd4, %r6, 4, %rd2;
    ld.global.f32 %f2, [%rd3];
    ld.global.f32 %f3, [%rd4];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r6, %r6, 32;
    bra loop;
finish:
    {warp_reduce('%f1', 'add')}
    mov.u32 %r6, %laneid;
    setp.ne.u32 %p3, %r6, 0;
    @%p3 ret;
    mul.f32 %f1, %f1, %a1;
    mad.wide.u32 %rd3, %r3, 4, %d3;
    st.global.f32 [%rd3], %f1;
    ret;
""", batched="states scores")


BLOCK_MIX = entry("block_mix", {"u64": "states scores out", "u32": "width count"}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.y;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mul.lo.u32 %r4, %r3, %n2;
    mad.wide.u32 %rd1, %r4, 4, %d2;
    mov.b32 %f1, 0fFF800000;
    mov.u32 %r5, 0;
max_loop:
    mad.wide.u32 %rd2, %r5, 4, %rd1;
    ld.global.f32 %f2, [%rd2];
    max.f32 %f1, %f1, %f2;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p2, %r5, %n2;
    @%p2 bra max_loop;
    mov.u32 %r5, 0;
    mov.b32 %f3, 0;
    mov.b32 %f4, 0;
mix_loop:
    mad.wide.u32 %rd2, %r5, 4, %rd1;
    ld.global.f32 %f2, [%rd2];
    sub.f32 %f2, %f2, %f1;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f2;
    add.f32 %f3, %f3, %f2;
    mad.lo.u32 %r6, %r5, %n1, %r1;
    mad.wide.u32 %rd3, %r6, 4, %d1;
    ld.global.f32 %f5, [%rd3];
    fma.rn.f32 %f4, %f5, %f2, %f4;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p2, %r5, %n2;
    @%p2 bra mix_loop;
    div.rn.f32 %f4, %f4, %f3;
    mad.lo.u32 %r6, %r3, %n1, %r1;
    mad.wide.u32 %rd3, %r6, 4, %d3;
    st.global.f32 [%rd3], %f4;
    ret;
""", batched="states scores out")


CONVOLUTION = entry("causal_conv", {
    "u64": "x static_weight dynamic_weight history out",
    "u32": "width taps position head_dim history_slots", "f32": "scale",
}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mov.u32 %r11, %ctaid.z;
    add.u32 %n3, %n3, %r11;
    mad.wide.u32 %rd1, %r1, 4, %d1;
    ld.global.f32 %f1, [%rd1];
    rem.u32 %r3, %n3, %n5;
    mad.lo.u32 %r4, %r3, %n1, %r1;
    mad.wide.u32 %rd2, %r4, 4, %d4;
    st.global.f32 [%rd2], %f1;
    setp.ne.u64 %p2, %d3, 0;
    @%p2 mad.wide.u32 %d3, %r11, %batch_stride, %d3;
    mov.u32 %r5, 0;
loop:
    // Read in-chunk lags directly from the input rows. Those CTAs may not have
    // written their history slots yet.
    setp.gt.u32 %p5, %r5, %r11;
    @%p5 bra cached_lag;
    mul.wide.u32 %rd5, %r5, %batch_stride;
    sub.u64 %rd2, %rd1, %rd5;
    bra load_lag;
cached_lag:
    sub.u32 %r7, %n3, %r5;
    rem.u32 %r7, %r7, %n5;
    mad.lo.u32 %r4, %r7, %n1, %r1;
    mad.wide.u32 %rd2, %r4, 4, %d4;
load_lag:
    ld.global.f32 %f2, [%rd2];
    sub.u32 %r8, %n2, 1;
    sub.u32 %r8, %r8, %r5;
    selp.u32 %r8, %r5, %r8, %p2;
    mad.lo.u32 %r4, %r8, %n1, %r1;
    mad.wide.u32 %rd3, %r4, 4, %d2;
    ld.global.f32 %f3, [%rd3];
    @!%p2 bra accumulate;
    div.u32 %r9, %r1, %n4;
    mad.lo.u32 %r4, %r9, %n2, %r5;
    mad.wide.u32 %rd3, %r4, 4, %d3;
    ld.global.f32 %f4, [%rd3];
    add.f32 %f3, %f3, %f4;
accumulate:
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p3, %r5, %n2;
    setp.le.u32 %p4, %r5, %n3;
    and.pred %p3, %p3, %p4;
    @%p3 bra loop;
    mul.f32 %f1, %f1, %a1;
    mad.wide.u32 %rd4, %r1, 4, %d5;
    st.global.f32 [%rd4], %f1;
    ret;
""", batched="x out")


ATTENTION_SCORES = entry("attention_scores", {
    "u64": "q k_cache relative relative_project scores",
    "u32": "window head_dim groups_per_kv pos capacity rel_width rel_extent",
    "f32": "scale",
}, f"""
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.y;
    mov.u32 %r10, %ctaid.z;
    add.u32 %n4, %n4, %r10;
    add.u32 %r10, %n4, 1;
    min.u32 %n1, %n1, %r10;
    shr.u32 %r4, %r2, 5;
    mad.lo.u32 %r4, %r1, 4, %r4;
    setp.ge.u32 %p1, %r4, %n1;
    @%p1 ret;
    div.u32 %r5, %r3, %n3;
    mul.lo.u32 %r6, %r3, %n2;
    mad.wide.u32 %rd1, %r6, 4, %d1;
    sub.u32 %r7, %n4, %r4;
    rem.u32 %r7, %r7, %n5;
    mad.lo.u32 %r8, %r5, %n5, %r7;
    mul.lo.u32 %r8, %r8, %n2;
    mad.wide.u32 %rd2, %r8, 4, %d2;
    mov.u32 %r9, %laneid;
    mov.b32 %f1, 0;
k_loop:
    setp.ge.u32 %p2, %r9, %n2;
    @%p2 bra k_done;
    mad.wide.u32 %rd3, %r9, 4, %rd1;
    mad.wide.u32 %rd4, %r9, 4, %rd2;
    ld.global.f32 %f2, [%rd3];
    ld.global.f32 %f3, [%rd4];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r9, %r9, 32;
    bra k_loop;
k_done:
    mul.f32 %f1, %f1, %a1;
    setp.ge.u32 %p3, %r4, %n7;
    @%p3 bra finish;
    mul.lo.u32 %r6, %r3, %n6;
    mul.lo.u32 %r7, %r4, %n6;
    mad.wide.u32 %rd1, %r6, 4, %d3;
    mad.wide.u32 %rd2, %r7, 4, %d4;
    mov.u32 %r9, %laneid;
rel_loop:
    setp.ge.u32 %p2, %r9, %n6;
    @%p2 bra finish;
    mad.wide.u32 %rd3, %r9, 4, %rd1;
    mad.wide.u32 %rd4, %r9, 4, %rd2;
    ld.global.f32 %f2, [%rd3];
    ld.global.f32 %f3, [%rd4];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r9, %r9, 32;
    bra rel_loop;
finish:
    {warp_reduce('%f1', 'add')}
    mov.u32 %r9, %laneid;
    setp.ne.u32 %p4, %r9, 0;
    @%p4 ret;
    mad.lo.u32 %r6, %r3, %n1, %r4;
    mad.wide.u32 %rd3, %r6, 4, %d5;
    st.global.f32 [%rd3], %f1;
    ret;
""", batched="q relative scores")


ATTENTION_MIX = entry("attention_mix", {
    "u64": "scores v_cache sink out",
    "u32": "window head_dim groups_per_kv pos capacity",
}, f"""
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r9, %ctaid.z;
    add.u32 %n4, %n4, %r9;
    add.u32 %r9, %n4, 1;
    min.u32 %n1, %n1, %r9;
    mul.lo.u32 %r3, %r2, %n1;
    mad.wide.u32 %rd1, %r3, 4, %d1;
    mad.wide.u32 %rd2, %r2, 4, %d3;
    ld.global.f32 %f1, [%rd2];
    mov.f32 %f9, %f1;
    mov.u32 %r4, %r1;
max_loop:
    setp.ge.u32 %p1, %r4, %n1;
    @%p1 bra max_done;
    mad.wide.u32 %rd3, %r4, 4, %rd1;
    ld.global.f32 %f2, [%rd3];
    max.f32 %f1, %f1, %f2;
    add.u32 %r4, %r4, 128;
    bra max_loop;
max_done:
    {block_reduce('%f1', 'max', 'attention_max')}
    mov.b32 %f3, 0;
    mov.u32 %r4, %r1;
sum_loop:
    setp.ge.u32 %p1, %r4, %n1;
    @%p1 bra sum_done;
    mad.wide.u32 %rd3, %r4, 4, %rd1;
    ld.global.f32 %f2, [%rd3];
    sub.f32 %f2, %f2, %f1;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f2;
    st.global.f32 [%rd3], %f2;
    add.f32 %f3, %f3, %f2;
    add.u32 %r4, %r4, 128;
    bra sum_loop;
sum_done:
    {block_reduce('%f3', 'add', 'attention_sum')}
    sub.f32 %f9, %f9, %f1;
    mul.f32 %f9, %f9, 0f3FB8AA3B;
    ex2.approx.f32 %f9, %f9;
    add.f32 %f3, %f3, %f9;
    div.u32 %r5, %r2, %n3;
    mul.lo.u32 %r5, %r5, %n5;
    mov.u32 %r6, %r1;
outputs:
    setp.ge.u32 %p2, %r6, %n2;
    @%p2 ret;
    mov.b32 %f4, 0;
    mov.u32 %r4, 0;
vals:
    sub.u32 %r7, %n4, %r4;
    rem.u32 %r7, %r7, %n5;
    add.u32 %r8, %r5, %r7;
    mad.lo.u32 %r8, %r8, %n2, %r6;
    mad.wide.u32 %rd4, %r8, 4, %d2;
    mad.wide.u32 %rd3, %r4, 4, %rd1;
    ld.global.f32 %f5, [%rd4];
    ld.global.f32 %f6, [%rd3];
    fma.rn.f32 %f4, %f5, %f6, %f4;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p3, %r4, %n1;
    @%p3 bra vals;
    div.rn.f32 %f4, %f4, %f3;
    mad.lo.u32 %r8, %r2, %n2, %r6;
    mad.wide.u32 %rd5, %r8, 4, %d4;
    st.global.f32 [%rd5], %f4;
    add.u32 %r6, %r6, 128;
    bra outputs;
""", REDUCTION, batched="scores sink out")


KV_STORE = entry("kv_store", {
    "u64": "key value keys values",
    "u32": "width head_dim capacity pos",
}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mov.u32 %r5, %ctaid.z;
    add.u32 %r5, %r5, %n4;
    rem.u32 %n4, %r5, %n3;
    div.u32 %r3, %r1, %n2;
    rem.u32 %r4, %r1, %n2;
    mad.lo.u32 %r3, %r3, %n3, %n4;
    mad.lo.u32 %r3, %r3, %n2, %r4;
    mad.wide.u32 %rd1, %r1, 4, %d1;
    mad.wide.u32 %rd2, %r1, 4, %d2;
    mad.wide.u32 %rd3, %r3, 4, %d3;
    mad.wide.u32 %rd4, %r3, 4, %d4;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    st.global.f32 [%rd3], %f1;
    st.global.f32 [%rd4], %f2;
    ret;
""", batched="key value")


ROUTE = entry("select_experts", {
    "u64": "router_logits correction probabilities expert_ids",
    "u32": "experts top_k", "f32": "route_scale",
}, f"""
    mov.u32 %r1, %tid.x;
    mov.u64 %rd1, dynamic_shared;
    mad.wide.u32 %rd2, %n1, 4, %rd1;
    mov.u32 %r2, %r1;
init:
    setp.ge.u32 %p1, %r2, %n1;
    @%p1 bra initialized;
    mad.wide.u32 %rd3, %r2, 4, %d1;
    mad.wide.u32 %rd4, %r2, 4, %d2;
    ld.global.f32 %f1, [%rd3];
    ld.global.f32 %f2, [%rd4];
    {sigmoid('%f1', '%f3')}
    add.f32 %f2, %f2, %f3;
    mad.wide.u32 %rd5, %r2, 4, %rd1;
    mad.wide.u32 %rd6, %r2, 4, %rd2;
    st.shared.f32 [%rd5], %f2;
    st.shared.f32 [%rd6], %f3;
    add.u32 %r2, %r2, 128;
    bra init;
initialized:
    bar.sync 0;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 ret;
    mov.u32 %r5, 0;
    mov.b32 %f6, 0;
next_expert:
    mov.u32 %r2, 0;
    mov.u32 %r6, 0;
    mov.b32 %f4, 0fFF800000;
scan:
    mad.wide.u32 %rd5, %r2, 4, %rd1;
    ld.shared.f32 %f2, [%rd5];
    setp.gt.f32 %p2, %f2, %f4;
    @%p2 mov.f32 %f4, %f2;
    @%p2 mov.u32 %r6, %r2;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p2, %r2, %n1;
    @%p2 bra scan;
    mad.wide.u32 %rd5, %r6, 4, %rd1;
    mov.b32 %f4, 0fFF800000;
    st.shared.f32 [%rd5], %f4;
    mad.wide.u32 %rd6, %r6, 4, %rd2;
    ld.shared.f32 %f5, [%rd6];
    mad.wide.u32 %rd7, %r5, 4, %d3;
    st.global.f32 [%rd7], %f5;
    add.f32 %f6, %f6, %f5;
    mad.wide.u32 %rd8, %r5, 4, %d4;
    st.global.u32 [%rd8], %r6;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p2, %r5, %n2;
    @%p2 bra next_expert;
    add.f32 %f6, %f6, 0f1E3CE508;
    div.rn.f32 %f6, %a1, %f6;
    mov.u32 %r5, 0;
normalize:
    mad.wide.u32 %rd7, %r5, 4, %d3;
    ld.global.f32 %f5, [%rd7];
    mul.f32 %f5, %f5, %f6;
    st.global.f32 [%rd7], %f5;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p2, %r5, %n2;
    @%p2 bra normalize;
    ret;
""", batched="router_logits probabilities expert_ids")


SWIGLU = entry("softcapped_swiglu", {
    "u64": "gate_up out", "u32": "width", "f32": "gate_cap up_cap",
}, f"""
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.y;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mad.lo.u32 %r3, %r3, %n1, %r1;
    mov.u32 %r4, %ctaid.y;
    mad.lo.u32 %r4, %r4, %n1, %r3;
    add.u32 %r5, %r4, %n1;
    mad.wide.u32 %rd1, %r4, 4, %d1;
    mad.wide.u32 %rd2, %r5, 4, %d1;
    mad.wide.u32 %rd3, %r3, 4, %d2;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    {sigmoid('%f1', '%f3')}
    div.rn.f32 %f1, %f1, %a1;
    div.rn.f32 %f2, %f2, %a2;
    tanh.approx.f32 %f1, %f1;
    tanh.approx.f32 %f2, %f2;
    mul.f32 %f1, %f1, %a1;
    mul.f32 %f2, %f2, %a2;
    mul.f32 %f1, %f1, %f2;
    mul.f32 %f1, %f1, %f3;
    st.global.f32 [%rd3], %f1;
    ret;
""", batched="gate_up out")


MIX_EXPERTS = entry("mix_experts", {
    "u64": "expert_values probabilities out", "u32": "width top_k",
}, """
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r1, %r1, 128, %r2;
    setp.ge.u32 %p1, %r1, %n1;
    @%p1 ret;
    mov.u32 %r3, 0;
    mov.b32 %f1, 0;
loop:
    mad.lo.u32 %r4, %r3, %n1, %r1;
    mad.wide.u32 %rd1, %r4, 4, %d1;
    mad.wide.u32 %rd2, %r3, 4, %d2;
    ld.global.f32 %f2, [%rd1];
    ld.global.f32 %f3, [%rd2];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p2, %r3, %n2;
    @%p2 bra loop;
    mad.wide.u32 %rd3, %r1, 4, %d3;
    st.global.f32 [%rd3], %f1;
    ret;
""", batched="expert_values probabilities out")


HEAD_SCATTER = vector("head_scatter", "bucket order out", "cap", """
    add.u64 %rd2, %d1, %rd1;
    add.u64 %rd3, %d2, %rd1;
    ld.global.f32 %f1, [%rd2];
    ld.global.u32 %r2, [%rd3];
    div.rn.f32 %f1, %f1, %a1;
    tanh.approx.f32 %f1, %f1;
    mul.f32 %f1, %f1, %a1;
    mad.wide.u32 %rd4, %r2, 4, %d3;
    st.global.f32 [%rd4], %f1;
""")


ARGMAX = entry("argmax", {
    "u64": "values indices out_values out_indices invalid",
    "u32": "count groups",
}, """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, 128, %r1;
    mul.lo.u32 %r4, %n2, 128;
    mov.b32 %f1, 0fFF800000;
    mov.u32 %r5, -1;
    mov.u32 %r13, 0;
    setp.eq.u64 %p2, %d2, 0;
scan:
    setp.ge.u32 %p1, %r3, %n1;
    @%p1 bra store;
    mad.wide.u32 %rd1, %r3, 4, %d1;
    ld.global.f32 %f2, [%rd1];
    mov.b32 %r6, %f2;
    and.b32 %r6, %r6, 0x7FFFFFFF;
    setp.ge.u32 %p3, %r6, 0x7F800000;
    @%p3 mov.u32 %r13, 1;
    @%p3 bra next;
    mov.u32 %r6, %r3;
    @%p2 bra compare;
    mad.wide.u32 %rd2, %r3, 4, %d2;
    ld.global.u32 %r6, [%rd2];
compare:
    setp.gt.f32 %p3, %f2, %f1;
    setp.eq.f32 %p4, %f2, %f1;
    setp.lt.u32 %p5, %r6, %r5;
    and.pred %p4, %p4, %p5;
    or.pred %p3, %p3, %p4;
    @%p3 mov.f32 %f1, %f2;
    @%p3 mov.u32 %r5, %r6;
next:
    add.u32 %r3, %r3, %r4;
    bra scan;
store:
    setp.ne.u32 %p3, %r13, 0;
    @%p3 atom.global.or.b32 %r14, [%d5], 1;
    mov.u64 %rd3, max_values;
    mov.u64 %rd4, max_ids;
    mad.wide.u32 %rd5, %r1, 4, %rd3;
    mad.wide.u32 %rd6, %r1, 4, %rd4;
    st.shared.f32 [%rd5], %f1;
    st.shared.u32 [%rd6], %r5;
    bar.sync 0;
    mov.u32 %r7, 64;
reduce:
    setp.ge.u32 %p6, %r1, %r7;
    @%p6 bra sync;
    add.u32 %r8, %r1, %r7;
    mad.wide.u32 %rd7, %r8, 4, %rd3;
    mad.wide.u32 %rd8, %r8, 4, %rd4;
    ld.shared.f32 %f2, [%rd7];
    ld.shared.u32 %r6, [%rd8];
    setp.gt.f32 %p3, %f2, %f1;
    setp.eq.f32 %p4, %f2, %f1;
    setp.lt.u32 %p5, %r6, %r5;
    and.pred %p4, %p4, %p5;
    or.pred %p3, %p3, %p4;
    @%p3 mov.f32 %f1, %f2;
    @%p3 mov.u32 %r5, %r6;
    @%p3 st.shared.f32 [%rd5], %f1;
    @%p3 st.shared.u32 [%rd6], %r5;
sync:
    bar.sync 0;
    shr.u32 %r7, %r7, 1;
    setp.ne.u32 %p3, %r7, 0;
    @%p3 bra reduce;
    setp.ne.u32 %p6, %r1, 0;
    @%p6 ret;
    mad.wide.u32 %rd1, %r2, 4, %d3;
    mad.wide.u32 %rd2, %r2, 4, %d4;
    st.global.f32 [%rd1], %f1;
    st.global.u32 [%rd2], %r5;
    ret;
""", ".shared .align 4 .b32 max_values[128], max_ids[128];")


def source():
    return "".join((
        HEADER, ADD, CAST, ATTN_GATE, MESSAGE, normalize("rms_norm"), normalize("clip_norm", clip=True),
        dense("dense_matvec"), dense("message_down", split_bf16=True),
        ROTATE, PACKED_ROW, BLOCK_SCORES, BLOCK_MIX, CONVOLUTION,
        ATTENTION_SCORES, ATTENTION_MIX, KV_STORE, ROUTE, SWIGLU, MIX_EXPERTS, HEAD_SCATTER, ARGMAX,
    ))
