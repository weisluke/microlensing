#!/usr/bin/env python3
"""Register/spill/stack numbers of the shooting kernels from build.log files written with -Xptxas -v."""
import re
import sys
from pathlib import Path

TARGET = re.compile(r"^=== (\S+) ===$")
ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
USED = re.compile(r"Used (\d+) registers")
SMEM = re.compile(r"(\d+) bytes smem")


def parse(text, pattern="shoot_cells"):
    rows, current, target = [], None, None
    for line in text.splitlines():
        m = TARGET.match(line.strip())
        if m:
            target = m.group(1)
            continue
        m = ENTRY.search(line)
        if m:
            current = {"target": target, "function": m.group(1), "sm": m.group(2)} if pattern in m.group(1) else None
            continue
        if current is None:
            continue
        m = STACK.search(line)
        if m:
            current.update(stack_bytes=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
            continue
        m = USED.search(line)
        if m:
            smem = SMEM.search(line)
            current.update(registers=int(m.group(1)), smem_bytes=int(smem.group(1)) if smem else 0)
            rows.append(current)
            current = None
    return rows


def main(argv=None):
    for build_dir in (argv or sys.argv[1:]):
        log = Path(build_dir) / "build.log"
        print(f"{build_dir}:")
        for r in parse(log.read_text()):
            print(f"  {r['target']:16s} regs {r['registers']:3d}  stack {r.get('stack_bytes', 0):5d} B  "
                  f"spills {r.get('spill_stores', 0)}/{r.get('spill_loads', 0)} B  smem {r['smem_bytes']} B  {r['function'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
