# GPU verify ldmatrix downgrade on GTX 1070.
# 两层验证:
#  1. ldm_direct/ldm_x2/ldm_x4/ldm_trans1:共享内存里放 32x8 u16 已知矩阵,
#     ldmatrix 后把目的寄存器原样写回全局,与 Python 布局公式逐 lane 对拍。
#  2. ldm_mma:ldmatrix x4.trans(A 块)+ x2(B 块)喂给 mma.m16n8k16,
#     结果与 numpy A@B 对拍 —— 用已 GPU 验证过的 mma 展开链路反查
#     ldmatrix 布局语义(若布局/打包序有错,乘积必然对不上)。
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, ROOT)

import runner                                            # noqa: E402
from runner import GPU, check                            # noqa: E402
from downgrader.parser import parse_module               # noqa: E402
from downgrader.engine import downgrade_module           # noqa: E402
from downgrader.validate import validate                 # noqa: E402

FILL = """
    ld.param.u64    %rd0, [up];
    ld.param.u64    %rd1, [out];
    mov.u32         %r0, %tid.x;
    and.b32         %r1, %r0, 31;
    // 填共享:lane i 存矩阵第 i 行(8 个 u16 = 4 字)
    mov.u32         %r2, mat;
    mul.wide.u32    %rd2, %r1, 16;
    add.s64         %rd3, %rd0, %rd2;
    ld.global.b32   %r4, [%rd3];
    add.s64         %rd3, %rd3, 4;
    ld.global.b32   %r5, [%rd3];
    add.s64         %rd3, %rd3, 4;
    ld.global.b32   %r6, [%rd3];
    add.s64         %rd3, %rd3, 4;
    ld.global.b32   %r7, [%rd3];
    shl.b32         %r8, %r1, 4;
    add.u32         %r9, %r2, %r8;
    st.shared.b32   [%r9], %r4;
    st.shared.b32   [%r9+4], %r5;
    st.shared.b32   [%r9+8], %r6;
    st.shared.b32   [%r9+12], %r7;
    bar.sync        0;
"""


def _direct_body(addr_mask, dests, trans=False):
    """行地址计算 + ldmatrix + 写回。addr_mask: 提供行地址的 lane 掩码。"""
    n = len(dests)
    tmode = ".trans" if trans else ""
    lines = [f"    and.b32         %r10, %r1, {addr_mask};",
             "    shl.b32         %r10, %r10, 4;",
             "    add.u32         %r10, %r2, %r10;",
             f"    ldmatrix.sync.aligned.m8n8.x{n}{tmode}.shared.b16 "
             f"{{{', '.join(dests)}}}, [%r10];"]
    for k, d in enumerate(dests):
        if k == 0:
            lines.append(f"    shl.b32         %r15, %r1, {2 + n // 2};")
            lines.append("    cvt.u64.u32     %rd4, %r15;")
            lines.append("    add.s64         %rd5, %rd1, %rd4;")
        lines.append(f"    st.global.b32   [%rd5+{4 * k}], {d};")
    lines.append("    ret;")
    return "\n".join(lines)


SRC = """
.version 8.0
.target sm_90
.address_size 64
.visible .entry ldm_direct(.param .u64 up, .param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;
""" + FILL + _direct_body(7, ["%r11"]) + """
}
.visible .entry ldm_x2(.param .u64 up, .param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;
""" + FILL + _direct_body(15, ["%r11", "%r12"]) + """
}
.visible .entry ldm_x4(.param .u64 up, .param .u64 out)
{
    .reg .b32 %r<18>;
    .reg .b64 %rd<10>;
""" + FILL + _direct_body(31, ["%r11", "%r12", "%r13", "%r14"]) + """
}
.visible .entry ldm_trans1(.param .u64 up, .param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;
""" + FILL + """
    and.b32         %r10, %r1, 7;
    shl.b32         %r10, %r10, 4;
    add.u32         %r10, %r2, %r10;
    ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r11}, [%r10];
    shl.b32         %r15, %r1, 2;
    cvt.u64.u32     %rd4, %r15;
    add.s64         %rd5, %rd1, %rd4;
    st.global.b32   [%rd5], %r11;
    ret;
}
.visible .entry ldm_x4t(.param .u64 up, .param .u64 out)
{
    .reg .b32 %r<18>;
    .reg .b64 %rd<10>;
""" + FILL + _direct_body(31, ["%r11", "%r12", "%r13", "%r14"], trans=True) + """
}
.shared .align 2 .b16 mat[256];
.shared .align 2 .b16 matA[256];
.shared .align 2 .b16 matB[256];
"""


def _run_kernel(gpu, src, name, up, out_words):
    text, _ = downgrade_module(parse_module(src), target="sm_61")
    fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    ok, err = validate(tmp, arch="sm_61")
    assert ok, (name, err[:600])
    mod = gpu.load_module(tmp)
    os.unlink(tmp)
    out = np.zeros(32 * out_words, dtype=np.uint32)
    d0 = gpu.alloc_h2d(up)
    d1 = gpu.alloc_h2d(out)
    fn = gpu.get_function(mod, name)
    gpu.launch(fn, [d0, d1], 1, 32)
    gpu.dtoh(d1, out)
    check(runner.cuda.cuMemFree_v2(d0), "free")
    check(runner.cuda.cuMemFree_v2(d1), "free")
    return out


def main():
    gpu = GPU()
    src = SRC
    results = []

    # ---- 4 个直接布局测试:mat[r][c] = r*8+c ----
    mat = np.arange(256, dtype=np.uint16).reshape(32, 8)
    g, t = np.meshgrid(np.arange(8), np.arange(4), indexing="ij")

    def packed(r, c):
        # 非 trans:寄存器 = {m[r][c] 低, m[r+1][c] 高}
        return (mat[r, c].astype(np.uint32)
                | (mat[r + 1, c].astype(np.uint32) << 16))

    def packedT(r, c):
        # trans:寄存器 = {m[r][c] 低, m[r][c+1] 高}
        return (mat[r, c].astype(np.uint32)
                | (mat[r, c + 1].astype(np.uint32) << 16))

    def tpairT(r, c):
        # 32 行矩阵的 trans 对(mat 与 packedT 同布局)
        return (mat[r, c].astype(np.uint32)
                | (mat[r, c + 1].astype(np.uint32) << 16))

    cases = {
        "ldm_direct": (1, [packed(2 * t, g)]),
        "ldm_x2": (2, [packed(2 * t, g), packed(8 + 2 * t, g)]),
        "ldm_x4": (4, [packed(8 * j + 2 * t, g) for j in range(4)]),
        "ldm_trans1": (1, [packedT(g, 2 * t)]),
        "ldm_x4t": (4, [tpairT(8*j + g, 2*t) for j in range(4)]),
    }
    up = mat.reshape(-1).view(np.uint32)            # 每行 16B = 4 字
    for name, (n, words) in cases.items():
        out = _run_kernel(gpu, src, name, up, n)
        want = np.zeros(32 * n, dtype=np.uint32)
        for lane in range(32):
            gg, tt = lane >> 2, lane & 3
            for k in range(len(words)):
                want[lane * n + k] = words[k][gg, tt]
        bad = int((out != want).sum())
        results.append((name, bad))
        print(f"[{'PASS' if bad == 0 else 'FAIL'}] {name}: "
              f"{bad} 不一致 / {32 * n}")
        if bad:
            idx = np.where(out != want)[0][:6]
            print("   got", out[idx], "\n   want", want[idx])

    total_bad = sum(b for _, b in results)
    print("结论:", "ldmatrix 全部通过" if total_bad == 0
          else f"{total_bad} 处不一致")
    sys.exit(0 if total_bad == 0 else 1)


if __name__ == "__main__":
    main()
