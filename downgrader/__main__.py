# CLI: python -m downgrader input.ptx [-o output.ptx]
import argparse
import os
import re
import sys

# Windows consoles default to legacy codepages (cp1252/cp936) that cannot
# encode the CJK status output below; force UTF-8 everywhere.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from .extract import extract_ptx, split_modules
from .parser import parse_module, UnsupportedError
from .engine import downgrade_module
from .validate import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _safe_name(s):
    """从不可信文本(cuobjdump 输出的 kernel 名)生成安全的文件名成分。"""
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s).strip("._") or "kernel"
    return s[:80]


def _safe_out(p, in_dir):
    """输出路径:规范化后必须落在输入文件目录或项目目录内。"""
    p = os.path.realpath(os.path.abspath(p))
    in_dir = os.path.realpath(os.path.abspath(in_dir))
    if not (p == in_dir or p.startswith(in_dir + os.sep)
            or p == ROOT or p.startswith(ROOT + os.sep)):
        sys.exit(f"输出路径必须在输入文件目录或项目目录内: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(
        description="PTX 降级器:把高代架构 PTX 重写为低代目标可加载的形式")
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--gui", action="store_true",
                    help="启动图形界面(免命令行,开箱即用)")
    ap.add_argument("--target", default="sm_61",
                    help="降级目标架构:sm_61 (Pascal) / sm_70 / sm_75 / "
                         "sm_80 / sm_86 (Ampere, 30系) / sm_89 (Ada)")
    args = ap.parse_args()

    if args.gui or args.input is None:
        from .gui import main as gui_main
        return gui_main()

    # .ptx 当文本读;.dll/.so/.cubin/.fatbin 先用 cuobjdump 提取内嵌 PTX
    ext = os.path.splitext(args.input)[1].lower()
    in_dir = os.path.dirname(os.path.realpath(os.path.abspath(args.input)))
    if ext == ".ptx":
        src = open(args.input, encoding="utf-8", errors="replace").read()
        name = _safe_name(os.path.splitext(os.path.basename(args.input))[0])
        jobs = [(name, src)]
    else:
        try:
            blobs = extract_ptx(args.input)
        except RuntimeError as e:
            print(f"[提取失败] {e}", file=sys.stderr)
            sys.exit(1)
        if not blobs:
            print("[未找到 PTX] 该二进制里没有内嵌 PTX 段(纯 SASS 构建?)",
                  file=sys.stderr)
            sys.exit(1)
        jobs = []
        for arch, text in blobs:
            for i, mtext in enumerate(split_modules(text)):
                jobs.append((_safe_name(f"{arch}_kernel{i}"), mtext))
        print(f"[提取] {args.input}: {len(jobs)} 个 PTX kernel")
        if not jobs:
            return

    suffix = "sm" + args.target[3:]
    for name, src in jobs:
        try:
            mod = parse_module(src)
            out_text, warns = downgrade_module(mod, target=args.target)
        except UnsupportedError as e:
            print(f"[不支持的指令] {e}", file=sys.stderr)
            sys.exit(2)

        default_out = os.path.join(in_dir, f"{name}.{suffix}.ptx")
        out_path = _safe_out(args.output or default_out, in_dir)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        for w in warns:
            print(f"[改写] {w}")

        if not args.no_validate:
            ok, err = validate(out_path, arch=args.target)
            if ok:
                print(f"[ptxas] {out_path}: 校验通过")
            else:
                print(f"[ptxas] 校验失败:\n{err}", file=sys.stderr)
                sys.exit(3)
        print(f"[完成] {out_path}")


if __name__ == "__main__":
    main()
