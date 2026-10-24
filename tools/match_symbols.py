#!/usr/bin/env python3
"""Recover XDK symbol addresses for one XBE by matching another title's function.

Both titles link the same XDK, so the statically linked library functions are the same
code even when the sections are laid out differently. Given the reference XBE (where the
symbol is known) and this one, take the reference function's instruction shapes --
register operands kept, image addresses and branch targets masked -- and find the
candidate functions whose shapes match exactly.

Identical thunks collide by shape alone, so the candidates are then resolved by order:
a section keeps its library link order across titles, so the assignment of reference
symbols (sorted by address) to candidates has to be increasing.

    python3 tools/match_symbols.py <ref.xbe> <ref_analysis.json> <this.xbe> <this_analysis.json> \
        <this_functions.json> SYMBOL=0xADDR ... | @argfile
"""
import json
import sys

from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_OP_REG, CS_OP_IMM, CS_OP_MEM

IMAGE_LO, IMAGE_HI = 0x00010000, 0x00800000


def sections(analysis_json):
    return json.load(open(analysis_json))["sections"]


def hx(v):
    return int(v, 16) if isinstance(v, str) else int(v)


def read_va(data, secs, va, n):
    for s in secs:
        lo = hx(s["virtual_addr"])
        if lo <= va < lo + hx(s["raw_size"]):
            off = hx(s["raw_addr"]) + (va - lo)
            return data[off:off + n]
    return None


def section_of(secs, va):
    for s in secs:
        lo = hx(s["virtual_addr"])
        if lo <= va < lo + hx(s["virtual_size"]):
            return s["name"]
    return None


def shapes(md, data, va, limit=400):
    out = []
    for insn in md.disasm(data, va):
        ops = []
        for op in insn.operands:
            if op.type == CS_OP_REG:
                ops.append(("r", insn.reg_name(op.reg)))
            elif op.type == CS_OP_IMM:
                v = op.imm & 0xFFFFFFFF
                ops.append(("i", "A" if IMAGE_LO <= v < IMAGE_HI else v))
            elif op.type == CS_OP_MEM:
                m = op.mem
                d = m.disp & 0xFFFFFFFF
                ops.append(("m",
                            insn.reg_name(m.base) if m.base else "-",
                            insn.reg_name(m.index) if m.index else "-",
                            m.scale,
                            "A" if IMAGE_LO <= d < IMAGE_HI else d))
        out.append((insn.mnemonic, tuple(ops)))
        if insn.mnemonic.startswith("ret") or len(out) >= limit:
            break
    return out


def main():
    ref_xbe, ref_json, this_xbe, this_json, funcs_json = sys.argv[1:6]
    ref = open(ref_xbe, "rb").read()
    this = open(this_xbe, "rb").read()
    ref_secs, this_secs = sections(ref_json), sections(this_json)
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True

    raw_funcs = json.load(open(funcs_json))
    if isinstance(raw_funcs, dict):
        raw_funcs = raw_funcs["functions"]
    funcs = sorted(int(f.get("start", f.get("address")), 16) for f in raw_funcs)

    wanted = []
    for arg in sys.argv[6:]:
        if arg.startswith("@"):
            for line in open(arg[1:]):
                line = line.split("#")[0].strip()
                if line:
                    name, _, addr = line.partition("=")
                    wanted.append((name.strip(), int(addr, 16)))
            continue
        name, _, addr = arg.partition("=")
        wanted.append((name, int(addr, 16)))
    wanted.sort(key=lambda s: s[1])

    # Export thunks are only reachable through the export table, so a call graph misses
    # them; index the sections by their first four bytes as well and let the shape decide.
    by_prefix = {}
    for s in this_secs:
        lo, size = hx(s["virtual_addr"]), hx(s["raw_size"])
        blob = this[hx(s["raw_addr"]):hx(s["raw_addr"]) + size]
        for o in range(0, max(0, size - 4)):
            by_prefix.setdefault((s["name"], blob[o:o + 4]), []).append(lo + o)

    by_shape = {}
    for va in funcs:
        sec = section_of(this_secs, va)
        raw = read_va(this, this_secs, va, 2048)
        if raw is None:
            continue
        key = (sec, tuple(shapes(md, raw, va)))
        by_shape.setdefault(key, []).append(va)

    cands = []
    for name, ref_va in wanted:
        raw = read_va(ref, ref_secs, ref_va, 2048)
        sec = section_of(ref_secs, ref_va)
        if raw is None or sec is None:
            cands.append((name, ref_va, sec, None, []))
            continue
        want = tuple(shapes(md, raw, ref_va))
        hits = set(by_shape.get((sec, want), []))
        for va in by_prefix.get((sec, raw[:4]), []):
            if va in hits:
                continue
            got = read_va(this, this_secs, va, 2048)
            if got is not None and tuple(shapes(md, got, va)) == want:
                hits.add(va)
        cands.append((name, ref_va, sec, sec, sorted(hits)))

    # library link order survives across titles, so the assignment has to be increasing
    # within a section: longest chain over (symbol, candidate) pairs
    best = {}  # (section, candidate) -> (count, parent key)
    order = {}
    for name, ref_va, sec, sec2, hits in cands:
        if sec2 is None:
            continue
        prev = [(k, v) for k, v in best.items() if k[0] == sec and k[1] < (hits[0] if hits else 0)]
        for c in hits:
            count, parent = 1, None
            for (k, (n, p)) in best.items():
                if k[0] == sec and k[1] < c and n + 1 > count:
                    count, parent = n + 1, k
            key = (sec, c)
            if key not in best or best[key][0] < count:
                best[key] = (count, parent)
                order[key] = (name, ref_va)

    # walk back from the longest chain
    end = max(best.items(), key=lambda kv: kv[1][0])[0] if best else None
    picked = {}
    while end is not None:
        picked[end] = order[end]
        end = best[end][1]

    exact = 0
    for name, ref_va, sec, sec2, hits in cands:
        if sec2 is None:
            print(f"{name:38s} ref {ref_va:08X} not in a section")
            continue
        if not hits:
            print(f"{name:38s} ref {sec} {ref_va:08X}  ->  no match")
            continue
        got = [c for c in hits if (sec, c) in picked and picked[(sec, c)][0] == name]
        if len(hits) == 1:
            exact += 1
        note = "" if len(hits) == 1 else f"  ({len(hits)} same shape)"
        if got:
            print(f"{name:38s} ref {sec} {ref_va:08X}  ->  0x{got[0]:08X}{note}")
        else:
            print(f"{name:38s} ref {sec} {ref_va:08X}  ->  unresolved, candidates "
                  + ", ".join(f"0x{c:08X}" for c in hits))
    print(f"# {exact} of {len(wanted)} unique by shape alone", file=sys.stderr)


main()
