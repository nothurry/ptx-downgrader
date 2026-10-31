# GPU verify redux.sync downgrade (shfl butterfly) on GTX 1070.
# u32 六种 op(v = lane+1)+ s32 三种 op(v = lane-16),逐 lane 与 numpy 对拍:
# 硬件语义 = 全 warp 规约,所有 lane 结果一致。
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

SRC = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry redux_u32(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<4>;
    ld.param.u64    %rd0, [out];
    mov.u32         %r0, %tid.x;
    add.u32         %r1, %r0, 1;        // v = lane+1
    redux.sync.add.u32 %r2,  %r1, 0xffffffff;
    redux.sync.min.u32 %r3,  %r1, 0xffffffff;
    redux.sync.max.u32 %r4,  %r1, 0xffffffff;
    redux.sync.and.b32 %r5,  %r1, 0xffffffff;
    redux.sync.or.b32  %r6,  %r1, 0xffffffff;
    redux.sync.xor.b32 %r7,  %r1, 0xffffffff;
    mul.wide.u32    %rd1, %r0, 24;
    add.s64         %rd2, %rd0, %rd1;
    st.global.b32   [%rd2],    %r2;
    st.global.b32   [%rd2+4],  %r3;
    st.global.b32   [%rd2+8],  %r4;
    st.global.b32   [%rd2+12], %r5;
    st.global.b32   [%rd2+16], %r6;
    st.global.b32   [%rd2+20], %r7;
    ret;
}
.visible .entry redux_s32(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64    %rd0, [out];
    mov.u32         %r0, %tid.x;
    add.s32         %r1, %r0, -16;      // v = lane-16
    redux.sync.add.s32 %r2, %r1, 0xffffffff;
    redux.sync.min.s32 %r3, %r1, 0xffffffff;
    redux.sync.max.s32 %r4, %r1, 0xffffffff;
    mul.wide.u32    %rd1, %r0, 12;
    add.s64         %rd2, %rd0, %rd1;
    st.global.b32   [%rd2],    %r2;
    st.global.b32   [%rd2+4],  %r3;
    st.global.b32   [%rd2+8],  %r4;
    ret;
}
"""


def _run(gpu, name, out_words, dtype):
    text, _ = downgrade_module(parse_module(SRC), target="sm_61")
    fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    ok, err = validate(tmp, arch="sm_61")
    assert ok, (name, err[:600])
    mod = gpu.load_module(tmp)
    os.unlink(tmp)
    out = np.zeros(32 * out_words, dtype=dtype)
    d0 = gpu.alloc_h2d(out)
    fn = gpu.get_function(mod, name)
    gpu.launch(fn, [d0], 1, 32)
    gpu.dtoh(d0, out)
    check(runner.cuda.cuMemFree_v2(d0), "free")
    return out


def main():
    gpu = GPU()
    v = np.arange(1, 33, dtype=np.uint32)
    want = np.array([v.sum(), v.min(), v.max(),
                     np.bitwise_and.reduce(v), np.bitwise_or.reduce(v),
                     np.bitwise_xor.reduce(v)], dtype=np.uint32)
    out = _run(gpu, "redux_u32", 6, np.uint32).reshape(32, 6)
    bad = int((out != want).sum())
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] redux u32 "
          f"(add/min/max/and/or/xor): {bad} 不一致 / 192")
    if bad:
        ops = ["add", "min", "max", "and", "or", "xor"]
        for j in range(6):
            if (out[:, j] != want[j]).any():
                print(f"    {ops[j]}: got {out[0, j]} want {want[j]}")

    vs = np.arange(-16, 16, dtype=np.int32)
    wants = np.array([vs.sum(), vs.min(), vs.max()], dtype=np.int32)
    outs = _run(gpu, "redux_s32", 3, np.int32).reshape(32, 3)
    bads = int((outs != wants).sum())
    print(f"[{'PASS' if bads == 0 else 'FAIL'}] redux s32 "
          f"(add/min/max): {bads} 不一致 / 96")
    if bads:
        ops = ["add", "min", "max"]
        for j in range(3):
            if (outs[:, j] != wants[j]).any():
                print(f"    {ops[j]}: got {outs[0, j]} want {wants[j]}")

    total = bad + bads
    print("结论:", "redux 全部通过" if total == 0 else f"{total} 处不一致")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
