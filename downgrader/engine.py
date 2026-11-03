# 降级规则引擎:parse 后的模块 → 一组 pass → 可被 ptxas -arch=sm_61 接受的模块
# 每个 pass 都可以做"一变多"(cp.async 展开),展开出的指令继承原谓词。
import re

from .parser import (Inst, Label, UnsupportedError, render_module,   # noqa: F401
                     brace_flatten)
from .mma import pass_mma

# ---- 特殊寄存器 ------------------------------------------------------------
# 值为位宽:PTX ISA 7.0 只允许它们作为 mov/cvt 等的源,ALU 操作数必须先物化。
SPECIAL_REGS = {
    "%tid.x": 32, "%tid.y": 32, "%tid.z": 32,
    "%ntid.x": 32, "%ntid.y": 32, "%ntid.z": 32,
    "%ctaid.x": 32, "%ctaid.y": 32, "%ctaid.z": 32,
    "%nctaid.x": 32, "%nctaid.y": 32, "%nctaid.z": 32,
    "%warpid": 32, "%nwarpid": 32, "%smid": 32, "%nsmid": 32,
    "%laneid": 32, "%clock": 32,
    "%clock64": 64, "%gridid": 64,
    "%lanemask_eq": 32, "%lanemask_lt": 32, "%lanemask_le": 32,
    "%lanemask_gt": 32, "%lanemask_ge": 32,
    "%dynamic_smem_size": 32, "%total_smem_size": 32,
    "%totsmem": 32, "%envreg": 32,
}
SPECIAL_RE = re.compile(
    r"%((?:tid|ntid|ctaid|nctaid)\.[xyz]|warpid|nwarpid|smid|nsmid|laneid|"
    r"clock|clock64|gridid|lanemask_(?:eq|lt|le|gt|ge)|dynamic_smem_size|"
    r"total_smem_size)\b")

# ---- 目标架构 ----------------------------------------------------------
# 降级目标不再是硬编码的 sm_61:任何"应用 PTX 目标 > 本机目标"的组合都适用。
# 规则按【指令引入代】分级,目标代 >= 引入代的指令保持原样。
#   sm_61: Pascal     sm_70: Volta    sm_75: Turing
#   sm_80/86: Ampere  sm_89: Ada      sm_90: Hopper
SUPPORTED_TARGETS = ("sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89")

_ARCH_ORDER = {f"sm_{n}": n for n in
               (61, 70, 75, 80, 86, 89)}


def _arch_num(target):
    return _ARCH_ORDER.get(target, 999)


# 引入代 >= 目标代时拒绝/降级的指令(键 = opcode 前缀,值 = (引入代, 说明))。
# 引入代 < 目标代的指令由硬件原生支持,直接透传。
REJECT_BY_ARCH = {
    "wgmma": (90, "Hopper wgmma:需要张量内存加速器"),
    "tcgen05": (100, "Blackwell tcgen05:第5代张量核指令"),
    "st.async": (90, "sm_90 异步存储"),
    "ld.async": (90, "sm_90 异步加载"),
    "red.async": (90, "sm_90 异步归约"),
    "st.bulk": (100, "sm_100 批量存储"),
    "multimem": (90, "NVLink 多播访存"),
    "mapa": (90, "cluster 地址映射:需要 thread block cluster"),
    "clusterlaunchcontrol": (100, "cluster 启动控制"),
    "barrier.cluster": (90, "cluster 屏障:需要 thread block cluster"),
    "setmaxnreg": (90, "Hopper 寄存器重分配"),
    "elect.sync": (90, "warp 选举:展开需解析双目的 {pred, reg} 形式,暂未支持"),
    "tensormap": (90, "TMA 张量描述符"),
    "cp.async.bulk": (90, "TMA 大批量异步拷贝:需要 TMA 硬件"),
    "cp.reduce.async.bulk": (90, "TMA 归约拷贝:需要 TMA 硬件"),
    "mbarrier.try_wait": (80, "mbarrier 非阻塞等待返回谓词,暂无安全降级"),
    "wmma": (70, "sm_70 旧式张量 API:无降级,请改用 mma.sync 或手写 FMA"),
}

# 所有目标一律拒绝(即使目标代达标也无意义,如向 sm_61 降 sm_75 指令时
# 这些指令的目标代本来就高于目标——放在这里只为报错文案统一)。
REJECT_ANY = {
    "stmatrix": (90, "sm_90 fragment 回写:降级需重排布局"),
}

REJECT_SUBSTR = (".cluster", "cta_group", ".shared::cta", "::cta")


class TempAlloc:
    """每入口的临时寄存器分配器:追加独立命名的声明向量,避免与原寄存器冲突。
    get("32u") 返回 .u32 声明的寄存器(shl/shr 的移位量要求 .u32 类型)。"""
    def __init__(self):
        self.counters = {}

    def get(self, bits):
        base = {32: "%rdg", "32u": "%rgu", 64: "%rddg",
                "pred": "%rgp"}[bits]
        n = self.counters.get(bits, 0)
        self.counters[bits] = n + 1
        return f"{base}{n}"

    def decls(self):
        out = []
        if self.counters.get(32):
            out.append(f".reg .b32 %rdg<{self.counters[32]}>;")
        if self.counters.get("32u"):
            out.append(f".reg .u32 %rgu<{self.counters['32u']}>;")
        if self.counters.get("64"):
            out.append(f".reg .b64 %rddg<{self.counters[64]}>;")
        if self.counters.get("pred"):
            out.append(f".reg .pred %rgp<{self.counters['pred']}>;")
        return out


def _emit(prefix_inst, lines, out):
    """把 (opcode, operands) 行展开进 out,继承原指令的谓词。"""
    for op, operands in lines:
        out.append(Inst(prefix_inst.pred, op, operands, prefix_inst.line_no))


def pass_reject(entry, target="sm_61"):
    tgt = _arch_num(target)
    for x in entry.body:
        if not isinstance(x, Inst):
            continue
        base = x.opcode.split(".")[0]
        for table in (REJECT_BY_ARCH, REJECT_ANY):
            if base in table:
                intro, why = table[base]
                if tgt < intro:
                    raise UnsupportedError(
                        f"{entry.name} 行{x.line_no}: `{x.opcode}` — {why} "
                        f"(sm_{tgt} 目标不支持,指令为 sm_{intro} 引入)")
                break
        else:
            for bad in REJECT_SUBSTR:
                if bad in x.opcode:
                    raise UnsupportedError(
                        f"{entry.name} 行{x.line_no}: `{x.opcode}` — "
                        f"包含 cluster/TMA 相关限定,目标架构无等价物")


def pass_opcode_fix(entry):
    """版本内安全的小改写:dp4a 限定符、mov 立即数类型、缓存提示剥离。"""
    for x in entry.body:
        if not isinstance(x, Inst):
            continue
        op = x.opcode
        if op.startswith("dp4a."):
            parts = op.split(".")
            if parts[1] in ("lo", "hi"):        # dp4a 没有 lo/hi(dp2a 才有)
                if parts[1] == "hi":
                    raise UnsupportedError(
                        f"{entry.name} 行{x.line_no}: dp4a.hi 无对应语义")
                op = "dp4a." + ".".join(parts[2:])
        if op == "mov.f16":
            op = "mov.b16"                       # mov.f16 不接受立即数
        elif op == "mov.f16x2":
            op = "mov.b32"
        op = re.sub(r"\.L[123]::\w+", "", op)    # 缓存逐出提示:仅提示,可剥离
        x.opcode = op


def _bracket_parts(operand):
    """'[base+off]' → (base, off);'[base]' → (base, 0)。"""
    assert operand.startswith("[") and operand.endswith("]"), operand
    inner = operand[1:-1]
    if "+" in inner:
        base, off = inner.split("+", 1)
        return base, int(off)
    return inner, 0


def pass_cpasync(entry, tmp):
    """cp.async → ld.global + st.shared(同步化);commit 组丢弃;
    wait_group/wait_all → bar.sync 0(保守但正确)。"""
    body = []
    for x in entry.body:
        if not isinstance(x, Inst):
            body.append(x)
            continue
        base = x.opcode.split(".")[0]
        if (x.opcode.startswith("cp.async") and "bulk" not in x.opcode
                and not x.opcode.startswith(("cp.async.commit_group",
                                             "cp.async.wait_group"))):
            if len(x.operands) not in (3, 4):
                raise UnsupportedError(
                    f"{entry.name} 行{x.line_no}: cp.async 操作数异常")
            dst_b, src_b = x.operands[0], x.operands[1]
            cp_size = int(x.operands[2].strip(), 0)
            src_size = cp_size
            if len(x.operands) == 4:
                s = x.operands[3].strip()
                if not s.isdigit() or int(s) % 4 or int(s) > cp_size:
                    raise UnsupportedError(
                        f"{entry.name} 行{x.line_no}: "
                        f"cp.async src-size `{s}` 不支持")
                src_size = int(s)
            if cp_size not in (4, 8, 16) or cp_size % 4:
                raise UnsupportedError(
                    f"{entry.name} 行{x.line_no}: cp-size `{cp_size}` 不支持")
            dbase, doff = _bracket_parts(dst_b)
            sbase, soff = _bracket_parts(src_b)
            lines = []
            n_chunks = cp_size // 4
            src_chunks = src_size // 4
            for i in range(n_chunks):
                t = tmp.get(32)
                co, so = doff + 4 * i, soff + 4 * i
                if i < src_chunks:
                    lines.append(("ld.global.b32",
                                  [t, f"[{sbase}+{so}]" if so else f"[{sbase}]"]))
                else:                            # 超出 src-size 的部分零填充
                    lines.append(("mov.b32", [t, "0"]))
                lines.append(("st.shared.b32",
                              [f"[{dbase}+{co}]" if co else f"[{dbase}]", t]))
            _emit(x, lines, body)
        elif x.opcode in ("cp.async.commit_group",):
            pass                                  # 同步化后无需分组
        elif x.opcode in ("cp.async.wait_group", "cp.async.wait_all"):
            # wait 本是"本线程等本线程的拷贝";同步化后改为全员屏障。
            # 必须剥谓词:谓词化 bar 只让部分线程到齐 = 整块死锁。
            # 原语义下未执行 wait 的线程,其拷贝由块内其它屏障兜底,
            # 保守对齐不改变可见性结论(所有 st.shared 的消费点前
            # 原生代码也必有全量同步)。
            body.append(Inst(None, "bar.sync", ["0"], x.line_no))
        else:
            body.append(x)
    entry.body = body


def pass_mbarrier(entry):
    """mbarrier.init/arrive/inval → 丢弃;wait → bar.sync 0(保守正确)。"""
    body = []
    for x in entry.body:
        if not isinstance(x, Inst):
            body.append(x)
            continue
        if x.opcode.startswith("mbarrier."):
            if x.opcode.startswith("mbarrier.wait"):
                # 与 cp.async.wait 同理:谓词化 bar 会整块死锁,剥谓词
                body.append(Inst(None, "bar.sync", ["0"], x.line_no))
            elif x.opcode.startswith(("mbarrier.init", "mbarrier.inval")):
                pass
            elif x.opcode.startswith("mbarrier.arrive"):
                if len(x.operands) > 1:           # 有返回值 state
                    raise UnsupportedError(
                        f"{entry.name} 行{x.line_no}: "
                        f"mbarrier.arrive 的返回值被使用,无法丢弃")
                pass
            continue
        body.append(x)
    entry.body = body


def pass_softfloat(entry, tmp):
    """sm_75+ 数学指令的软件降级(sm_61 无硬件实现,ptxas 直接拒绝):
    tanh.approx.f32 d, a →
        mul.f32  t, a, 2.8853900817779268 (2·log2e)
        ex2.approx.f32 e, t
        add.f32  s, e, 1.0
        rcp.approx.f32 r, s          (r = 1/(1+2^(2x·log2e)))
        mul.f32  d, r, 2.0 → 减 1:tanh = 2r - 1
        实际公式:tanh(x) = 1 - 2/(e^(2x)+1) = 2/(e^(2x)+1) - 1 … 修正:
        r = 1/(1+E),  tanh = 1 - 2r·E/(1+?) … 采用标准恒等式:
        tanh(x) = (E-1)/(E+1), E = e^(2x)。用 r = 1/(E+1):
        tanh = (E+1-2)/(E+1) = 1 - 2·r。序列:
        mul.f32 t, a, 2.8853900817779268
        ex2.approx.f32 E, t
        add.f32  s, E, 1.0
        rcp.approx.f32 r, s
        fma.rn.f32 d, r, -2.0, 1.0      (d = 1 - 2r)
    精度:ex2/rcp 是 approx(约 22-23 bit),与硬件 tanh.approx 同级别。
    x→±Inf/NaN 边界:approx 指令遵循 IEEE 特殊值传递,极值行为与
    硬件 approx 有轻微差异(容差内)。
    bmsk.clamp.b32 d, a, b, c(位域掩码,sm_100)→
        语义:d = (a >> c) & ((1<<b)-1)(截断到 32 位,clamp 端点)
        降级:shl/sub/shr 三条。b/c 是 8 位立即数。"""
    body = []
    for x in entry.body:
        if not isinstance(x, Inst):
            body.append(x)
            continue
        if x.opcode in ("tanh.approx.f32", "tanh.f32", "tanh.approx.f16",
                        "tanh.f16"):
            if len(x.operands) != 2:
                body.append(x)
                continue
            d, a = x.operands[0], x.operands[1]
            f16src = x.opcode.endswith("f16")
            # f16 输入先转 f32 算完转回;f32 直接算
            t = tmp.get(32)
            e = tmp.get(32)
            s = tmp.get(32)
            r = tmp.get(32)
            pre = []
            a32 = a
            if f16src:
                h = tmp.get(32)
                pre.append(Inst(None, "cvt.f32.f16", [t, a], x.line_no))
                a32 = t
            pre.extend([
                Inst(None, "mul.f32",
                     [t, a32, "2.8853900817779268"], x.line_no),
                Inst(None, "ex2.approx.f32", [e, t], x.line_no),
                Inst(None, "add.f32", [s, e, "1.0"], x.line_no),
                Inst(None, "rcp.approx.f32", [r, s], x.line_no),
                Inst(None, "fma.rn.f32", [t, r, "-2.0", "1.0"], x.line_no),
            ])
            if f16src:
                d16 = tmp.get(32)
                pre.append(Inst(None, "cvt.rn.f16.f32", [d16, t], x.line_no))
                body.extend(pre)
                body.append(Inst(x.pred, "mov.b16", [d, d16], x.line_no))
            else:
                body.extend(pre)
                body.append(Inst(x.pred, "mov.f32", [d, t], x.line_no))
            continue
        if x.opcode.startswith("bmsk"):
            # 官方语义(PTX ISA 9.0 §9.7.1.23,ISA 7.6 / sm_70+ 引入):
            # bmsk.mode.b32 d, a, b —— 生成从位 a 起宽 b 的位掩码:
            #   a1 = a & 0x1f; mask0 = ~0 << a1
            #   b1 = b & 0x1f; sum = a1 + b1
            #   mask1 = ~0 << sum;若 sum 回绕(sum < a1)→ mask1 = 0
            #   .clamp 且 a >= 32 → mask0 = 0
            #   b1 == 0 → mask1 = ~0(由 shl 结果自然得出:~0 << 0 = ~0 ✓)
            #   d = mask0 & ~mask1
            # 降级序列(sm_61 原生指令,类型规则:移位 .b32/.u32 任意池,
            # 算术 sub.u32,逻辑 and.b32/setp 带 .u32):
            if len(x.operands) != 3:
                raise UnsupportedError(
                    f"{entry.name} 行{x.line_no}: bmsk 应为三操作数 "
                    f"d, a, b(官方语法 bmsk.mode.b32)")
            d, a, b = x.operands
            wrap = ".wrap" in x.opcode
            t = [tmp.get(32) for _ in range(4)]
            p = tmp.get("pred")
            seq = [
                Inst(x.pred, "and.b32", [t[0], a, "31"], x.line_no),    # a1
                Inst(x.pred, "mov.b32", [t[1], "-1"], x.line_no),
                Inst(x.pred, "shl.b32", [t[1], t[1], t[0]], x.line_no),  # mask0
                Inst(x.pred, "and.b32", [t[2], b, "31"], x.line_no),    # b1
                Inst(x.pred, "add.u32", [t[3], t[0], t[2]], x.line_no),  # sum
                # sum 回绕检测:无符号加法 sum < a1 即回绕 → mask1 = 0
                Inst(x.pred, "setp.lo.u32", [p, t[3], t[0]], x.line_no),
                Inst(x.pred, "selp.u32", [t[3], "0", "-1", p], x.line_no),
                # mask1 = mask1_sel << sum(回绕时 sel=0 → mask1=0;
                # b1==0 时 sum==a1 → ~0<<a1,后续 ~mask1 恰好补全 ✓)
                Inst(x.pred, "shl.b32", [t[3], t[3], t[3]], x.line_no),
                Inst(x.pred, "not.b32", [t[3], t[3]], x.line_no),        # ~mask1
                Inst(x.pred, "and.b32", [d, t[1], t[3]], x.line_no),     # d
            ]
            if not wrap:                          # .clamp:a >= 32 → mask0 = 0
                seq.insert(3, Inst(x.pred, "setp.hi.u32",
                                   [p, a, "31"], x.line_no))
                seq.insert(4, Inst(x.pred, "selp.u32",
                                   [t[1], "0", t[1], p], x.line_no))
            body.extend(seq)
            continue
        body.append(x)
    entry.body = body


def pass_griddepcontrol(entry):
    """sm_90 程序化依赖启动(PDL)剥除:
    - griddepcontrol.launch_dependents:提示"可以启动下一个 kernel"→ 无操作;
    - griddepcontrol.wait:等待前序 kernel 完成 → 保守替换为无操作
      (PDL 下 wait 之前的代码本就可以与前一 kernel 重叠执行;剥离后
      kernel 变成普通串行启动,语义安全,只损失启动重叠的收益)。
    """
    body = []
    for x in entry.body:
        if isinstance(x, Inst) and x.opcode.startswith("griddepcontrol."):
            continue
        body.append(x)
    entry.body = body


def _materialize_addr(operand, tmp, line_no):
    """处理 [base+off] 地址操作数:PTX 7.0 不允许 64 位基址 + 32 位寄存器
    偏移的混合宽度地址(实测 ptxas 与 JIT 都拒绝,特殊寄存器也同理)。
    对 32 位寄存器偏移(含特殊寄存器)统一:物化 → cvt.u64.u32 → add.s64,
    返回纯 [reg64] 形式。立即数偏移([base+8])是合法形态,不动。
    返回 (新操作数 或 None, 前置指令列表)。"""
    if not (operand.startswith("[") and operand.endswith("]")):
        return None, []
    inner = operand[1:-1]
    if "+" not in inner:
        return None, []
    base, off = inner.split("+", 1)
    off = off.strip()
    if re.fullmatch(r"\d+", off):          # 立即数偏移:合法,保留
        return None, []
    pre = []
    if SPECIAL_RE.fullmatch(off):
        t32 = tmp.get(32)
        pre.append(Inst(None, "mov.u32", [t32, off], line_no))
        src32 = t32
    elif re.fullmatch(r"%r\w*\d+", off):
        src32 = off                        # 已是 32 位普通寄存器
    else:
        return None, []
    t64off = tmp.get(64)
    pre.append(Inst(None, "cvt.u64.u32", [t64off, src32], line_no))
    tb = tmp.get(64)
    pre.append(Inst(None, "mov.u64", [tb, base.strip()], line_no))
    t64 = tmp.get(64)
    pre.append(Inst(None, "add.s64", [t64, tb, t64off], line_no))
    return f"[{t64}]", pre


def pass_special_regs(entry, tmp):
    """PTX ISA 7.0:特殊寄存器只能进 mov/cvt。三类形态都要物化:
    1. 纯操作数位置(add.u32 %r, %tid.x …);
    2. 地址算术内嵌([base+%tid.x]);
    3. 两者并存于同一指令。"""
    body = []
    for x in entry.body:
        if isinstance(x, Inst) and x.opcode.split(".")[0] != "mov":
            hit = False
            newops, pres = [], []
            for o in x.operands:
                o = o.strip()
                m = SPECIAL_RE.fullmatch(o)
                if m:
                    hit = True
                    t = tmp.get(SPECIAL_REGS["%" + m.group(1)])
                    pres.append(Inst(None, "mov.u32", [t, o], x.line_no))
                    newops.append(t)
                    continue
                no, pre = _materialize_addr(o, tmp, x.line_no)
                if no is not None:
                    hit = True
                    pres.extend(pre)
                    newops.append(no)
                else:
                    newops.append(o)
            if hit:
                body.extend(pres)
                x = Inst(x.pred, x.opcode, newops, x.line_no)
        body.append(x)
    entry.body = body


def pass_strip_cta(entry):
    """剥 opcode 里的 ::cta 作用域限定:非 cluster 启动下 ::cta ≡ 默认作用域。
    必须在 pass_reject 之前跑,否则带 .shared::cta 的行会被整行拒绝。"""
    for x in entry.body:
        if isinstance(x, Inst) and "::cta" in x.opcode:
            x.opcode = x.opcode.replace("::cta", "")


# ---- redux.sync → shfl 蝶形归约 -------------------------------------------
# redux.sync.op.type d, a, membermask 是 sm_80 引入的全 warp 规约。
# 全掩码时蝶形交换(offset 16,8,4,2,1)5 轮后所有 lane 拿到同一结果,
# 与硬件语义一致;子集掩码需要按 lane 有界交换,如实拒绝。
_REDUX_OPS = {
    "add.s32": "add.s32", "add.u32": "add.u32",
    "min.s32": "min.s32", "min.u32": "min.u32",
    "max.s32": "max.s32", "max.u32": "max.u32",
    "and.b32": "and.b32", "or.b32": "or.b32", "xor.b32": "xor.b32",
}


def pass_redux(entry, tmp, target="sm_61"):
    body = []
    for x in entry.body:
        if not isinstance(x, Inst) or not x.opcode.startswith("redux.sync"):
            body.append(x)
            continue
        if x.pred:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: 谓词化 redux 会使 shfl.sync "
                f"参与者不确定,不支持")
        key = x.opcode[len("redux.sync."):]
        if key not in _REDUX_OPS:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: `{x.opcode}` — "
                f"该 redux 变体无 shfl 等价降级")
        if len(x.operands) != 3:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: redux 操作数异常")
        d, a, mask = (o.strip() for o in x.operands)
        if mask not in ("0xffffffff", "4294967295", "-1"):
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: redux 子集掩码 `{mask}` "
                f"不支持(仅全 warp 0xffffffff)")
        op = _REDUX_OPS[key]
        if _arch_num(target) >= 70:
            shfl = "shfl.sync.bfly.b32"
        else:
            shfl = "shfl.bfly.b32"          # sm_70 前无 .sync 形式
        acc, t = tmp.get(32), tmp.get(32)
        emit = lambda o, *os_: body.append(                 # noqa: E731
            Inst(x.pred, o, list(os_), x.line_no))
        emit("mov.b32", acc, a)
        for off in (16, 8, 4, 2, 1):
            if shfl.endswith("sync.bfly.b32"):
                emit(shfl, t, acc, str(off), "31", "0xffffffff")
            else:
                emit(shfl, t, acc, str(off), "31")
            emit(op, acc, acc, t)
        emit("mov.b32", d, acc)
    entry.body = body


# ---- ldmatrix → 共享内存行地址交换 + 逐元素加载 ----------------------------
# ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16:硬件版由 lanes 0..8N-1
# 提供各 8x8 块的行地址,fragment 按 mma 布局分发到 32 lane。
# 展开:各 lane 把行地址(u32,共享地址恒 < 4GB)存进每 warp 的共享槽,
# bar.sync 后每 lane 按 mma fragment 布局取所需行的地址,读两个 u16 打包。
# 必须:① 全部用 b32 共享访存(u64 共享读写在本项目工具链上不可靠);
# ② 每条 ldmatrix 的读写块首尾各有 bar.sync —— 同一 entry 的多条
#    ldmatrix 复用同一批槽,尾屏障防止后一条的 store 与前一条跨 lane
#    的 read 竞争(三种实现变体 × GTX 1070 实测教训,详见 DEVLOG)。
# 布局(PTX ISA "ldmatrix",与 mma fragment 表一致,首元素 = 低半字):
#   非 trans:lane (g,t) 寄存器 j = {m_j[2t][g], m_j[2t+1][g]},行地址来自
#             lanes 8j+2t / 8j+2t+1
#   trans   :lane (g,t) 寄存器 j = {m_j[g][2t],  m_j[g][2t+1]},行地址来自
#             lane 8j+g
LDM_SHARED_VAR = "__dg_ldm"
LDM_TOTAL = 128 * 32                  # 每 warp 32 槽 × 4B × 32 warp
LDM_DECL_BASE = [".reg .b32 %dgla<8>;", ".reg .b32 %dglb<4>;",
                 ".reg .b16 %dglc<2>;"]

_REG_DECL_RE = re.compile(r"\.reg\s+(\.\w+)\s+(%\w+)(?:<(\d+)>)?\s*;")


def _reg_widths(entry):
    """扫描 entry 的 .reg 声明 → {寄存器名: 位宽}。池 %rd<N> 展开成员。"""
    widths = {}
    for x in entry.body:
        if isinstance(x, Inst) or not isinstance(x, str):
            continue
        m = _REG_DECL_RE.match(x.strip())
        if not m:
            continue
        bits = {"64": 64, "32": 32, "16": 16}.get(m.group(1)[2:], 32)
        if m.group(3):
            for i in range(int(m.group(3))):
                widths[f"{m.group(2)}{i}"] = bits
        else:
            widths[m.group(2)] = bits
    return widths


def _gen_ldmatrix(inst, widths, target="sm_61", k=0):
    """单条 ldmatrix 的 b32 共享交换展开(结构对齐已验证的 mma 交换)。
    k = 本条指令在 entry 内的序号:地址(%dglk<k>)与列偏移(%dglm<k>)寄存器
    按指令专属 —— 两条 ldmatrix 共用寄存器时,ptxas 会把后一条的寄存器
    定义提升进前一条的存取窗口(单线程合法),跨 lane 交换随即被污染。"""
    parts = inst.opcode.split(".")
    xn, trans = None, False
    for p in parts:
        if p in ("x1", "x2", "x4"):
            xn = int(p[1])
        elif p == "trans":
            trans = True
        elif p == "b4":
            raise UnsupportedError(
                f"行{inst.line_no}: ldmatrix .b4 子字形式不支持")
    if xn is None or parts[-1] != "b16":
        raise UnsupportedError(
            f"行{inst.line_no}: `{inst.opcode}` — 仅支持 m8n8.x1/x2/x4 .b16")
    if len(inst.operands) != 2:
        raise UnsupportedError(
            f"行{inst.line_no}: ldmatrix 操作数异常")
    dests = brace_flatten(inst.operands[0])
    if len(dests) != xn:
        raise UnsupportedError(
            f"行{inst.line_no}: ldmatrix 目的寄存器 {len(dests)} 个 != x{xn}")
    addr = inst.operands[1].strip()
    if not (addr.startswith("[") and addr.endswith("]")):
        raise UnsupportedError(
            f"行{inst.line_no}: ldmatrix 地址操作数异常: {addr}")
    inner = addr[1:-1].strip()
    if "+" in inner:
        base, off = inner.split("+", 1)
        if not off.lstrip("-").isdigit():
            raise UnsupportedError(
                f"行{inst.line_no}: ldmatrix 寄存器偏移不支持: {addr}")
        off = int(off, 0)
    else:
        base, off = inner, 0

    LA = [f"%dgla{i}" for i in range(8)]
    B = [f"%dglb{i}" for i in range(3)]
    H = [f"%dglc{i}" for i in range(2)]
    ADDR = f"%dglk{k}"                            # 本指令专属:行地址
    COLOFF = f"%dglm{k}"                          # 本指令专属:列偏移
    L = []

    def emit(op, *os_):
        L.append(Inst(inst.pred, op, list(os_), inst.line_no))

    # ---- 线性 tid → warp;lane/g/t ----
    emit("mov.u32", LA[0], "%tid.x")
    emit("mov.u32", LA[1], "%ntid.x")
    emit("mov.u32", LA[2], "%tid.y")
    emit("mov.u32", LA[3], "%ntid.y")
    emit("mov.u32", LA[4], "%tid.z")
    emit("mul.lo.u32", LA[5], LA[2], LA[1])
    emit("mul.lo.u32", LA[6], LA[1], LA[3])
    emit("mul.lo.u32", LA[7], LA[6], LA[4])
    emit("add.u32", LA[0], LA[0], LA[5])
    emit("add.u32", LA[0], LA[0], LA[7])          # 线性 tid
    emit("shr.u32", LA[1], LA[0], "5")            # warp(块内)
    emit("and.b32", LA[2], LA[0], "31")           # lane
    emit("shr.u32", LA[3], LA[2], "2")            # g
    emit("and.b32", LA[4], LA[2], "3")            # t

    # ---- 地址操作数 → u32(存入专属寄存器)----
    if base.startswith("%"):
        if widths.get(base) == 64:
            emit("cvt.u32.u64", ADDR, base)
        else:
            emit("mov.u32", ADDR, base)
    else:
        emit("mov.u32", ADDR, base)               # 共享变量符号
    if off:
        emit("add.u32", ADDR, ADDR, str(off))

    # ---- 全程 u32 共享寻址(共享地址恒 < 4GB;u64 地址链会触发
    #      ptxas 11.4 的调度问题,GTX 1070 实测)----
    # 槽基址 %dglb0 = __dg_ldm + warp*128;存本 lane 行地址
    emit("mov.u32", B[0], LDM_SHARED_VAR)
    emit("shl.b32", B[1], LA[1], "7")             # warp*128
    emit("add.u32", B[0], B[0], B[1])
    emit("shl.b32", B[1], LA[2], "2")             # lane*4
    emit("add.u32", B[2], B[0], B[1])
    emit("st.shared.u32", f"[{B[2]}]", ADDR)
    emit("bar.sync", "0")

    # ---- 列偏移(专属寄存器):非 trans 2g 字节;trans 4t 字节 ----
    if not trans:
        emit("shl.b32", LA[7], LA[3], "1")
    else:
        emit("shl.b32", LA[7], LA[4], "2")
    emit("mov.u32", COLOFF, LA[7])

    for j in range(xn):
        if not trans:
            # 行地址槽:8j+2t / 8j+2t+1 → 字节 32j+8t / +4
            emit("shl.b32", LA[7], LA[4], "3")
            emit("add.u32", LA[7], LA[7], str(32 * j))
        else:
            # 行地址槽:8j+g → 字节 32j+4g
            emit("shl.b32", LA[7], LA[3], "2")
            emit("add.u32", LA[7], LA[7], str(32 * j))
        emit("add.u32", B[1], B[0], LA[7])
        emit("ld.volatile.shared.u32", LA[5], f"[{B[1]}]")
        if not trans:
            emit("add.u32", LA[7], LA[7], "4")    # 下一行槽
            emit("add.u32", B[1], B[0], LA[7])
            emit("ld.volatile.shared.u32", LA[6], f"[{B[1]}]")
        emit("add.u32", B[1], LA[5], COLOFF)      # e0 = 行地址 + 列偏移
        emit("ld.volatile.shared.u16", H[0], f"[{B[1]}]")
        if not trans:
            emit("add.u32", B[1], LA[6], COLOFF)  # e1
        else:
            emit("add.u32", B[1], B[1], "2")      # e1 = e0 + 2
        emit("ld.volatile.shared.u16", H[1], f"[{B[1]}]")
        emit("mov.b32", dests[j], "{" + H[0] + ", " + H[1] + "}")
    # 读尾屏障:下一条 ldmatrix 的 store 不得越过它
    emit("bar.sync", "0")
    return L


def pass_ldmatrix(entry, target="sm_61"):
    body = []
    expanded = False
    widths = _reg_widths(entry)
    # 每条 ldmatrix 专属的地址/列偏移寄存器(池大小 = 指令数)
    n_ldm = sum(1 for x in entry.body
                if isinstance(x, Inst) and x.opcode.startswith("ldmatrix"))
    k = 0
    for x in entry.body:
        if not isinstance(x, Inst) or not x.opcode.startswith("ldmatrix"):
            body.append(x)
            continue
        if x.pred:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: 谓词化 ldmatrix 破坏 .aligned "
                f"全 warp 约定,不支持")
        body.extend(_gen_ldmatrix(x, widths, target, k))
        k += 1
        expanded = True
    if expanded:
        head = [f".shared .align 4 .b8 {LDM_SHARED_VAR}[{LDM_TOTAL}];"]
        head.extend(LDM_DECL_BASE)
        if n_ldm:
            head.append(f".reg .b32 %dglk<{n_ldm}>;")
            head.append(f".reg .b32 %dglm<{n_ldm}>;")
        entry.body = head + body
    else:
        entry.body = body


def downgrade_module(mod, target="sm_61"):
    """整体管线。target: 降级目标(sm_61/70/75/80/86/89)。
    返回 (渲染文本, warnings)。"""
    warns = []
    if target not in SUPPORTED_TARGETS:
        raise UnsupportedError(
            f"不支持的降级目标 {target};可选:{', '.join(SUPPORTED_TARGETS)}")
    # .version 按目标取目标代 ptxas 可接受的最低版本:
    #   sm_61/70/75 → 6.3,sm_80 → 7.0,sm_86/89 → 8.0
    # (实测 13.0 ptxas 报 "PTX .version 7.0 does not support .target sm_86")
    out_ver = ("6.3" if _arch_num(target) <= 75
               else ("7.0" if target == "sm_80" else "8.0"))
    if mod["version"] and float(mod["version"]) > float(out_ver):
        warns.append(f".version {mod['version']} → {out_ver}")
    mod["version"] = out_ver
    old_target = mod["target"]
    mod["target"] = target
    if old_target and old_target != target:
        warns.append(f".target {old_target} → {target}")

    tgt = _arch_num(target)
    for e in mod["entries"]:
        pass_strip_cta(e)             # ::cta ≡ 默认作用域,先剥再判
        pass_reject(e, target)
        tmp = TempAlloc()
        pass_opcode_fix(e)
        if tgt < 80:
            pass_cpasync(e, tmp)      # cp.async 是 sm_80 引入
        pass_mbarrier(e)
        pass_griddepcontrol(e)
        pass_softfloat(entry=e, tmp=tmp)
        pass_redux(e, tmp, target)    # redux.sync 是 sm_80 引入
        pass_ldmatrix(e, target)      # ldmatrix 是 sm_75 引入
        shared_used = [False]
        pass_mma(e, shared_used)
        pass_special_regs(e, tmp)
        if tmp.counters:
            e.body = tmp.decls() + e.body

    return render_module(mod), warns


def downgrade_text(ptx_text):
    from .parser import parse_module
    mod = parse_module(ptx_text)
    return downgrade_module(mod)
