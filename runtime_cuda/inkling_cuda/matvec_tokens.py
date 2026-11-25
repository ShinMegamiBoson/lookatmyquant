"""Generate the batched packed matvec kernels at the end of matvec.ptx.

`python -m runtime_cuda.inkling_cuda.matvec_tokens` rewrites everything from `// packed_matvec_tokens` onwards.

- `packed_matvec_tokens`: an unrouted projection for a batch of tokens (speculative checks, prefill chunks). A block
  serves a tile of 8 tokens, so each weight is decoded once and multiplied into every token's sum.
- `packed_matvec_routed_tokens`: a routed (expert) projection for a batch of tokens. Grid (row tiles, experts): each
  warp lists the (token, slot) pairs that selected its expert, then decodes each weight once per pass of 8, 4, 2 or 1
  pairs, sized to the pairs left.

Both keep packed_matvec's lanes, column order and reduction order for every token or pair, so each output is
bit-identical to the per-token kernel. The kernels are unrolled per token, which is why they are generated.
"""
from pathlib import Path

PTX = Path(__file__).with_name("matvec.ptx")


def _tokens():
    TILE = 8
    p = PTX
    src = p.read_bytes().decode().replace("\r\n", "\n")
    if "packed_matvec_tokens" in src:
        src = src[:src.index("\n// packed_matvec_tokens")] + "\n"

    acc = [f"%acc{t}" for t in range(TILE)]
    lines = [
        "",
        f"// packed_matvec_tokens: an unrouted projection for up to {TILE} tokens per block (grid.z tiles the tokens).",
        "// Each weight is decoded once and multiplied into every token's sum.",
        ".visible .entry packed_matvec_tokens(",
        "    .param .u64 x_ptr, .param .u64 codes_ptr, .param .u64 scales_ptr,",
        "    .param .u64 table_ptr, .param .u64 expert_ptr, .param .u64 y_ptr,",
        "    .param .u32 rows, .param .u32 width, .param .u32 words,",
        "    .param .u32 bits, .param .u32 group, .param .u32 hash_shift,",
        "    .param .u32 hash_multiplier, .param .u32 x_div, .param .u32 lloyd,",
        "    .param .u64 descriptors_ptr, .param .u64 expert_map_ptr, .param .u32 row_stride,",
        "    .param .u32 groups, .param .u32 group_rows, .param .u32 tokens",
        ")",
        "{",
        "    .reg .pred %p<9>;",
        f"    .reg .pred %pt<{TILE}>;",
        "    .reg .b32 %r<52>;",
        "    .reg .b64 %rd<24>;",
        "    .reg .f32 %f<9>;",
        f"    .reg .f32 %acc<{TILE}>;",
        f"    .reg .f32 %xv<{TILE}>;",
        "    .reg .b16 %h;",
        "",
        "    ld.param.u64 %rd1, [x_ptr];",
        "    ld.param.u64 %rd2, [codes_ptr];",
        "    ld.param.u64 %rd3, [scales_ptr];",
        "    ld.param.u64 %rd4, [table_ptr];",
        "    ld.param.u64 %rd5, [expert_ptr];",
        "    ld.param.u64 %rd6, [y_ptr];",
        "    ld.param.u32 %r1, [rows];",
        "    ld.param.u32 %r2, [width];",
        "    ld.param.u32 %r3, [words];",
        "    ld.param.u32 %r4, [bits];",
        "    ld.param.u32 %r5, [group];",
        "    ld.param.u32 %r6, [hash_shift];",
        "    ld.param.u32 %r7, [hash_multiplier];",
        "    ld.param.u32 %r9, [lloyd];",
        "    ld.param.u32 %r11, [row_stride];",
        "    ld.param.u32 %r42, [group_rows];",
        "    ld.param.u32 %r46, [tokens];",
        "    cvt.u64.u32 %rd18, %r11;",
        "",
        f"    // grid.z selects a tile of {TILE} tokens; %pt<t> marks the tokens that exist.",
        "    mov.u32 %r10, %ctaid.z;",
        f"    mul.lo.u32 %r47, %r10, {TILE};",
        "    sub.u32 %r46, %r46, %r47;",
        "    mul.wide.u32 %rd17, %r47, %r11;",
        "    add.u64 %rd1, %rd1, %rd17;",
        "    add.u64 %rd6, %rd6, %rd17;",
    ]
    for t in range(TILE):
        lines.append(f"    setp.gt.u32 %pt{t}, %r46, {t};")
    lines += [
        "",
        "    mov.u32 %r11, %ctaid.x;",
        "    mov.u32 %r12, %tid.x;",
        "    mov.u32 %r13, %laneid;",
        "    shr.u32 %r14, %r12, 5;",
        "    mad.lo.u32 %r15, %r11, 4, %r14;",
        "    mov.u32 %r43, %r15;",
        "    setp.ge.u32 %p1, %r15, %r1;",
        "    @%p1 bra DONE;",
        "    ld.global.s32 %r16, [%rd5];",
        "    setp.lt.s32 %p2, %r16, 0;",
        "    @%p2 bra DONE;",
        "    mad.lo.u32 %r17, %r16, %r42, %r43;",
        "    mul.lo.u32 %r18, %r17, %r3;",
        "    mad.wide.u32 %rd8, %r18, 4, %rd2;",
        "    mov.u64 %rd9, %rd1;",
        "    setp.ne.u32 %p3, %r9, 0;",
        "    setp.ne.u32 %p4, %r7, 0;",
    ]
    lines += [f"    mov.b32 {a}, 0;" for a in acc]
    lines += [
        "    mov.u32 %r21, %r13;",
        "LOOP:",
        "    setp.ge.u32 %p5, %r21, %r2;",
        "    @%p5 bra REDUCE;",
        "    sub.u32 %r22, %r5, 1;",
        "    and.b32 %r22, %r21, %r22;",
        "    sub.u32 %r23, %r21, %r22;",
        "    add.u32 %r23, %r23, %r5;",
        "    sub.u32 %r24, %r2, %r23;",
        "    selp.u32 %r24, %r21, %r24, %p3;",
        "    mul.lo.u32 %r25, %r24, %r4;",
        "    shr.u32 %r26, %r25, 5;",
        "    and.b32 %r27, %r25, 31;",
        "    mad.wide.u32 %rd10, %r26, 4, %rd8;",
        "    ld.global.u32 %r28, [%rd10];",
        "    add.u32 %r29, %r26, 1;",
        "    setp.lt.u32 %p6, %r29, %r3;",
        "    mov.u32 %r30, 0;",
        "    @%p6 ld.global.u32 %r30, [%rd10+4];",
        "    shr.u32 %r31, %r28, %r27;",
        "    sub.u32 %r32, 32, %r27;",
        "    shl.b32 %r33, %r30, %r32;",
        "    or.b32 %r34, %r31, %r33;",
        "    mov.u32 %r35, 1;",
        "    shl.b32 %r36, %r35, %r4;",
        "    sub.u32 %r36, %r36, 1;",
        "    selp.u32 %r37, %r36, 4095, %p3;",
        "    and.b32 %r38, %r34, %r37;",
        "    @%p4 mul.lo.u32 %r38, %r38, %r7;",
        "    @%p4 shr.u32 %r38, %r38, %r6;",
        "    mad.lo.u32 %r39, %r38, %r5, %r22;",
        "    mad.wide.u32 %rd11, %r39, 4, %rd4;",
        "    mad.wide.u32 %rd12, %r21, 4, %rd9;",
        "    ld.global.f32 %f2, [%rd11];",
    ]
    for t in range(TILE):
        lines += [
            f"    mov.b32 %xv{t}, 0;",
            f"    @%pt{t} ld.global.f32 %xv{t}, [%rd12];",
            f"    fma.rn.f32 %acc{t}, %f2, %xv{t}, %acc{t};",
            "    add.u64 %rd12, %rd12, %rd18;",
        ]
    lines += [
        "    add.u32 %r21, %r21, 32;",
        "    bra LOOP;",
        "REDUCE:",
    ]
    for t in range(TILE):
        for k in (16, 8, 4, 2, 1):
            lines += [f"    shfl.sync.bfly.b32 %f4, %acc{t}, {k}, 31, -1;", f"    add.f32 %acc{t}, %acc{t}, %f4;"]
    lines += [
        "    setp.ne.u32 %p7, %r13, 0;",
        "    @%p7 bra DONE;",
        "    mad.wide.u32 %rd13, %r17, 2, %rd3;",
        "    ld.global.b16 %h, [%rd13];",
        "    cvt.f32.f16 %f5, %h;",
        "    mad.wide.u32 %rd14, %r15, 4, %rd6;",
    ]
    for t in range(TILE):
        lines += [
            f"    mul.f32 %f6, %acc{t}, %f5;",
            f"    @%pt{t} st.global.f32 [%rd14], %f6;",
            "    add.u64 %rd14, %rd14, %rd18;",
        ]
    lines += ["DONE:", "    ret;", "}", ""]
    p.write_bytes((src.rstrip("\n") + "\n" + "\n".join(lines)).replace("\n", "\r\n").encode())


def _routed():
    TILE = 8
    MAX_PAIRS = 16  # tokens per call (PREFILL_CHUNK); a token selects an expert at most once
    p = PTX
    src = p.read_bytes().decode().replace("\r\n", "\n")
    if "packed_matvec_routed_tokens" in src:
        src = src[:src.index("\n// packed_matvec_routed_tokens")] + "\n"

    L = [
        "",
        f"// packed_matvec_routed_tokens: grid (row tiles, experts). Each block serves every (token, slot) of the batch",
        "// that selected its expert, decoding each weight once per pass of up to 8 pairs.",
        ".visible .entry packed_matvec_routed_tokens(",
        "    .param .u64 x_ptr, .param .u64 selected_ptr, .param .u64 y_ptr,",
        "    .param .u64 descriptors_ptr, .param .u64 expert_map_ptr,",
        "    .param .u32 rows, .param .u32 width, .param .u32 slots, .param .u32 x_div,",
        "    .param .u32 row_stride, .param .u32 groups, .param .u32 group_rows, .param .u32 tokens",
        ")",
        "{",
        "    .reg .pred %p<10>;",
        f"    .reg .pred %pv<{TILE}>;",
        "    .reg .b32 %r<64>;",
        "    .reg .b64 %rd<24>;",
        f"    .reg .b64 %xp<{TILE}>;",
        f"    .reg .b64 %yp<{TILE}>;",
        "    .reg .f32 %f<9>;",
        f"    .reg .f32 %acc<{TILE}>;",
        f"    .reg .f32 %xv<{TILE}>;",
        "    .reg .b16 %h;",
        f"    .shared .align 4 .b32 pairs[{4 * MAX_PAIRS}];",
        "",
        "    ld.param.u64 %rd1, [x_ptr];",
        "    ld.param.u64 %rd5, [selected_ptr];",
        "    ld.param.u64 %rd6, [y_ptr];",
        "    ld.param.u64 %rd15, [descriptors_ptr];",
        "    ld.param.u64 %rd16, [expert_map_ptr];",
        "    ld.param.u32 %r1, [rows];",
        "    ld.param.u32 %r2, [width];",
        "    ld.param.u32 %r50, [slots];",
        "    ld.param.u32 %r8, [x_div];",
        "    ld.param.u32 %r51, [row_stride];",
        "    ld.param.u32 %r44, [groups];",
        "    ld.param.u32 %r42, [group_rows];",
        "    ld.param.u32 %r52, [tokens];",
        "    cvt.u64.u32 %rd18, %r51;",
        "",
        "    mov.u32 %r53, %ctaid.y;",  # expert
        "    mov.u32 %r11, %ctaid.x;",
        "    mov.u32 %r12, %tid.x;",
        "    mov.u32 %r13, %laneid;",
        "    shr.u32 %r14, %r12, 5;",
        "    mad.lo.u32 %r15, %r11, 4, %r14;",
        "    setp.ge.u32 %p1, %r15, %r1;",
        "    @%p1 bra DONE;",
        "",
        "    // List this warp's (token, slot) pairs for the expert: token << 8 | slot.",
        "    mov.u32 %r54, 0;",  # count
        "    mul.lo.u32 %r55, %r52, %r50;",  # total entries
        "    mov.u32 %r56, 0;",  # base
        f"    mul.lo.u32 %r57, %r14, {4 * MAX_PAIRS};",
        "    mov.u32 %r58, pairs;",
        "    add.u32 %r57, %r57, %r58;",  # this warp's list (shared address)
        "    mov.u32 %r59, %lanemask_lt;",
        "SCAN:",
        "    setp.ge.u32 %p2, %r56, %r55;",
        "    @%p2 bra LISTED;",
        "    add.u32 %r20, %r56, %r13;",
        "    setp.lt.u32 %p3, %r20, %r55;",
        "    mov.u32 %r23, -1;",
        "    @!%p3 bra MATCH;",
        "    div.u32 %r21, %r20, %r50;",  # token
        "    mul.lo.u32 %r22, %r21, %r50;",
        "    sub.u32 %r22, %r20, %r22;",  # slot
        "    mul.wide.u32 %rd2, %r21, %r51;",
        "    add.u64 %rd2, %rd2, %rd5;",
        "    mad.wide.u32 %rd2, %r22, 4, %rd2;",
        "    ld.global.s32 %r23, [%rd2];",
        "MATCH:",
        "    setp.eq.s32 %p4, %r23, %r53;",
        "    vote.sync.ballot.b32 %r24, %p4, -1;",
        "    @!%p4 bra NEXT;",
        "    and.b32 %r25, %r24, %r59;",
        "    popc.b32 %r25, %r25;",
        "    add.u32 %r25, %r25, %r54;",
        "    shl.b32 %r26, %r21, 8;",
        "    or.b32 %r26, %r26, %r22;",
        "    mad.lo.u32 %r27, %r25, 4, %r57;",
        "    st.shared.u32 [%r27], %r26;",
        "NEXT:",
        "    popc.b32 %r24, %r24;",
        "    add.u32 %r54, %r54, %r24;",
        "    add.u32 %r56, %r56, 32;",
        "    bra SCAN;",
        "LISTED:",
        "    bar.warp.sync -1;",
        "    setp.eq.u32 %p2, %r54, 0;",
        "    @%p2 bra DONE;",
        "",
        "    // The expert's format and slab, as in packed_matvec's routed path.",
        "    mov.u32 %r16, %r53;",
        "    mov.u32 %r43, %r15;",
        "    setp.eq.u32 %p8, %r44, 1;",
        "    @%p8 bra LOOKUP;",
        "    div.u32 %r45, %r15, %r42;",
        "    rem.u32 %r43, %r15, %r42;",
        "    mad.lo.u32 %r16, %r16, %r44, %r45;",
        "LOOKUP:",
        "    mad.wide.u32 %rd17, %r16, 8, %rd16;",
        "    ld.global.v2.u32 {%r41, %r16}, [%rd17];",
        "    mad.wide.u32 %rd17, %r41, 48, %rd15;",
        "    ld.global.u64 %rd2, [%rd17];",
        "    ld.global.u64 %rd3, [%rd17+8];",
        "    ld.global.u64 %rd4, [%rd17+16];",
        "    ld.global.u32 %r3, [%rd17+24];",
        "    ld.global.u32 %r4, [%rd17+28];",
        "    ld.global.u32 %r5, [%rd17+32];",
        "    ld.global.u32 %r6, [%rd17+36];",
        "    ld.global.u32 %r7, [%rd17+40];",
        "    ld.global.u32 %r9, [%rd17+44];",
        "    mad.lo.u32 %r17, %r16, %r42, %r43;",
        "    mul.lo.u32 %r18, %r17, %r3;",
        "    mad.wide.u32 %rd8, %r18, 4, %rd2;",
        "    setp.ne.u32 %p3, %r9, 0;",
        "    setp.ne.u32 %p4, %r7, 0;",
        "    mul.lo.u32 %r60, %r2, 4;",  # bytes per x row
        "    mov.u32 %r61, 0;",  # first pair of this pass
        "PASS:",
        "    setp.ge.u32 %p2, %r61, %r54;",  # a pass may step past the end of the list
        "    @%p2 bra DONE;",
        "    sub.u32 %r62, %r54, %r61;",  # pairs left
        "    setp.gt.u32 %p2, %r62, 4;",
        "    @%p2 bra PASS8;",
        "    setp.gt.u32 %p2, %r62, 2;",
        "    @%p2 bra PASS4;",
        "    setp.gt.u32 %p2, %r62, 1;",
        "    @%p2 bra PASS2;",
        "    bra PASS1;",
    ]


    def emit_pass(T):
        out = [f"PASS{T}:"]
        for j in range(T):
            out += [
                f"    add.u32 %r20, %r61, {j};",
                f"    setp.lt.u32 %pv{j}, %r20, %r54;",
                f"    mov.b32 %acc{j}, 0;",
                f"    mov.u64 %xp{j}, %rd1;",
                f"    mov.u64 %yp{j}, %rd6;",
                f"    @!%pv{j} bra P{T}_PAIR{j}_DONE;",
                "    mad.lo.u32 %r27, %r20, 4, %r57;",
                "    ld.shared.u32 %r26, [%r27];",
                "    shr.u32 %r21, %r26, 8;",
                "    and.b32 %r22, %r26, 255;",
                "    mul.wide.u32 %rd2, %r21, %r51;",
                "    div.u32 %r28, %r22, %r8;",
                "    mul.wide.u32 %rd9, %r28, %r60;",
                f"    add.u64 %xp{j}, %rd1, %rd2;",
                f"    add.u64 %xp{j}, %xp{j}, %rd9;",
                "    mad.lo.u32 %r29, %r22, %r1, %r15;",
                "    mul.wide.u32 %rd9, %r29, 4;",
                f"    add.u64 %yp{j}, %rd6, %rd2;",
                f"    add.u64 %yp{j}, %yp{j}, %rd9;",
                f"P{T}_PAIR{j}_DONE:",
            ]
        out += [
            "    mov.u32 %r21, %r13;",
            f"P{T}_LOOP:",
            "    setp.ge.u32 %p5, %r21, %r2;",
            f"    @%p5 bra P{T}_REDUCE;",
            "    sub.u32 %r22, %r5, 1;",
            "    and.b32 %r22, %r21, %r22;",
            "    sub.u32 %r23, %r21, %r22;",
            "    add.u32 %r23, %r23, %r5;",
            "    sub.u32 %r24, %r2, %r23;",
            "    selp.u32 %r24, %r21, %r24, %p3;",
            "    mul.lo.u32 %r25, %r24, %r4;",
            "    shr.u32 %r26, %r25, 5;",
            "    and.b32 %r27, %r25, 31;",
            "    mad.wide.u32 %rd10, %r26, 4, %rd8;",
            "    ld.global.u32 %r28, [%rd10];",
            "    add.u32 %r29, %r26, 1;",
            "    setp.lt.u32 %p6, %r29, %r3;",
            "    mov.u32 %r30, 0;",
            "    @%p6 ld.global.u32 %r30, [%rd10+4];",
            "    shr.u32 %r31, %r28, %r27;",
            "    sub.u32 %r32, 32, %r27;",
            "    shl.b32 %r33, %r30, %r32;",
            "    or.b32 %r34, %r31, %r33;",
            "    mov.u32 %r35, 1;",
            "    shl.b32 %r36, %r35, %r4;",
            "    sub.u32 %r36, %r36, 1;",
            "    selp.u32 %r37, %r36, 4095, %p3;",
            "    and.b32 %r38, %r34, %r37;",
            "    @%p4 mul.lo.u32 %r38, %r38, %r7;",
            "    @%p4 shr.u32 %r38, %r38, %r6;",
            "    mad.lo.u32 %r39, %r38, %r5, %r22;",
            "    mad.wide.u32 %rd11, %r39, 4, %rd4;",
            "    ld.global.f32 %f2, [%rd11];",
            "    mul.wide.u32 %rd12, %r21, 4;",
        ]
        for j in range(T):
            out += [
                f"    mov.b32 %xv{j}, 0;",
                f"    add.u64 %rd13, %xp{j}, %rd12;",
                f"    @%pv{j} ld.global.f32 %xv{j}, [%rd13];",
                f"    fma.rn.f32 %acc{j}, %f2, %xv{j}, %acc{j};",
            ]
        out += ["    add.u32 %r21, %r21, 32;", f"    bra P{T}_LOOP;", f"P{T}_REDUCE:"]
        for j in range(T):
            for k in (16, 8, 4, 2, 1):
                out += [f"    shfl.sync.bfly.b32 %f4, %acc{j}, {k}, 31, -1;", f"    add.f32 %acc{j}, %acc{j}, %f4;"]
        out += [
            "    setp.ne.u32 %p7, %r13, 0;",
            f"    @%p7 bra P{T}_NEXT;",
            "    mad.wide.u32 %rd14, %r17, 2, %rd3;",
            "    ld.global.b16 %h, [%rd14];",
            "    cvt.f32.f16 %f5, %h;",
        ]
        for j in range(T):
            out += [f"    mul.f32 %f6, %acc{j}, %f5;", f"    @%pv{j} st.global.f32 [%yp{j}], %f6;"]
        out += [f"P{T}_NEXT:", f"    add.u32 %r61, %r61, {T};", "    bra PASS;"]
        return out


    for T in (8, 4, 2, 1):
        L += emit_pass(T)
    L += ["DONE:", "    ret;", "}", ""]
    p.write_bytes((src.rstrip("\n") + "\n" + "\n".join(L)).replace("\n", "\r\n").encode())


if __name__ == "__main__":
    _tokens()
    _routed()
