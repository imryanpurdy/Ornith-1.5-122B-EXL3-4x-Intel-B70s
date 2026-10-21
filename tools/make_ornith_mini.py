#!/usr/bin/env python3
"""Ornith-mini: a byte-exact 4-layer subset of an Ornith-1.5-122B EXL3 checkpoint, for 1-4 card tests.

The subset keeps, unchanged, every tensor of
  model.language_model.layers.{0,1,2,3}.*   (3 GDN layers + 1 full-attention layer, 256 experts each)
  model.language_model.embed_tokens.*, model.language_model.norm.*, lm_head.*
  mtp.*                                      (the whole MTP layer)
and drops layers 4.. and the vision tower (model.visual.*). The vision tower is not needed when the
mini is served with --language-model-only: vLLM then builds `visual` as a StageMissingLayer and its
weight loader skips it. Without that flag vLLM would build the tower and, because it does not track
loaded weights for quantized checkpoints, run it with uninitialized weights. Use --include-visual
for a copy that also works without --language-model-only.

config.json changes only text_config.num_hidden_layers and text_config.layer_types (truncated);
quantization_config.json keeps the tensor_storage entries of the kept modules; the index is rebuilt;
tokenizer, chat template, generation and processor files are copied byte for byte.

Subcommands
  create        copy the subset from --src into a new --dst (refuses an existing --dst)
  verify        re-derive the subset from --src and check --dst against it, tensor by tensor
  check-config  parse --dst (and --src) with transformers and vLLM; needs an image with vLLM,
                loads no weights and opens no device

create and verify use only the Python standard library. Safetensors headers are parsed directly and
tensor bytes are copied with os.pread/os.write as exact byte ranges: nothing is decoded or re-encoded.

Examples
  python3 tools/make_ornith_mini.py create --src "$MODEL_DIR" --dst "$MINI_DIR"
  python3 tools/make_ornith_mini.py verify --src "$MODEL_DIR" --dst "$MINI_DIR"
  python  tools/make_ornith_mini.py check-config --src "$MODEL_DIR" --dst "$MINI_DIR"
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

TOOL_VERSION = 1
NUM_LAYERS = 4
CHUNK = 64 << 20
MUL1_MARKER = 0x83DCD12D
EXL3_SUFFIXES = frozenset({"trellis", "suh", "svh", "mul1"})
ALLOWED_K = (4, 5, 6)
BLOCK = 128
PROVENANCE = "ornith_mini.json"

# Bit rates of the published checkpoint (model card + quantization_config): routed experts 4,
# attention / GDN / shared experts 5, lm_head 6 (head_bits), everything under mtp.* 4 (mtp_bits).
RECIPE_K = {"experts": 4, "dense": 5, "lm_head": 6, "mtp": 4}

# Layer 3 routed experts whose gate, up and down svh are all zero in the published weights:
# their output is exactly 0 whatever the trellis holds (exact-zero probes for the MoE tests).
DEAD_EXPERT_PROBES = {3: (72, 166, 180, 252)}

DTYPE_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2,
              "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}

# (source name, destination name, required). Everything else in the source directory is left out.
AUX_FILES = (
    ("tokenizer.json", "tokenizer.json", True),
    ("tokenizer_config.json", "tokenizer_config.json", True),
    ("vocab.json", "vocab.json", False),
    ("merges.txt", "merges.txt", False),
    ("tokenizer.model", "tokenizer.model", False),
    ("special_tokens_map.json", "special_tokens_map.json", False),
    ("added_tokens.json", "added_tokens.json", False),
    ("chat_template.jinja", "chat_template.jinja", False),
    ("chat_template.json", "chat_template.json", False),
    ("generation_config.json", "generation_config.json", True),
    ("preprocessor_config.json", "preprocessor_config.json", False),
    ("processor_config.json", "processor_config.json", False),
    ("video_preprocessor_config.json", "video_preprocessor_config.json", False),
    ("README.md", "SOURCE_README.md", False),  # the source model card (licence notice), verbatim
)
CHAT_TEMPLATES = ("chat_template.jinja", "chat_template.json")

_LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
_MTP_LAYER = re.compile(r"mtp\.layers\.(\d+)\.")
_EXPERT = re.compile(r"experts\.\d+\.")
GLOBAL_KEEP = ("model.language_model.embed_tokens.", "model.language_model.norm.", "lm_head.", "mtp.")
VISUAL = "model.visual."


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- checkpoint reading

@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple
    file: str     # shard basename
    begin: int    # absolute byte offset in the shard file
    nbytes: int


@dataclass
class Shard:
    file: str
    metadata: dict | None
    data_start: int
    file_size: int
    tensors: dict = field(default_factory=dict)   # name -> Tensor


@dataclass
class Checkpoint:
    path: str
    config: dict
    config_raw: bytes
    quant: dict
    quant_raw: bytes
    index: dict
    index_raw: bytes
    shards: dict          # basename -> Shard
    tensors: dict         # name -> Tensor, in index order
    problems: list        # structural problems found while reading


def _read_json(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), raw


def read_shard(path: str, problems: list) -> Shard:
    name = os.path.basename(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise Fail(f"{name}: too short for a safetensors file")
        (hlen,) = struct.unpack("<Q", prefix)
        if not 2 <= hlen <= min(size - 8, 1 << 30):
            raise Fail(f"{name}: implausible header length {hlen}")
        doc = json.loads(f.read(hlen))
    shard = Shard(file=name, metadata=doc.pop("__metadata__", None), data_start=8 + hlen, file_size=size)
    for tname, e in doc.items():
        dtype, shape = e["dtype"], tuple(e["shape"])
        if dtype not in DTYPE_SIZE:
            raise Fail(f"{name}: {tname}: unknown dtype {dtype}")
        b, end = e["data_offsets"]
        nbytes = DTYPE_SIZE[dtype] * math.prod(shape)
        if end - b != nbytes:
            problems.append(f"{name}: {tname}: data_offsets span {end - b} B, dtype x shape is {nbytes} B")
        shard.tensors[tname] = Tensor(tname, dtype, shape, name, shard.data_start + b, end - b)
    pos = shard.data_start
    for t in sorted(shard.tensors.values(), key=lambda t: t.begin):
        if t.begin != pos:
            problems.append(f"{name}: hole or overlap before {t.name} (offset {t.begin}, expected {pos})")
        pos = t.begin + t.nbytes
    if pos != size:
        problems.append(f"{name}: data ends at {pos}, file size is {size}")
    return shard


def load_checkpoint(path: str) -> Checkpoint:
    path = os.path.abspath(path)
    problems: list[str] = []
    config, config_raw = _read_json(os.path.join(path, "config.json"))
    quant, quant_raw = _read_json(os.path.join(path, "quantization_config.json"))
    index, index_raw = _read_json(os.path.join(path, "model.safetensors.index.json"))
    wmap = index["weight_map"]
    shards = {f: read_shard(os.path.join(path, f), problems) for f in sorted(set(wmap.values()))}
    stray = sorted(set(f for f in os.listdir(path) if f.endswith(".safetensors")) - set(shards))
    if stray:
        problems.append(f"safetensors files not referenced by the index: {stray}")
    tensors = {}
    for tname, f in wmap.items():
        t = shards[f].tensors.get(tname)
        if t is None:
            problems.append(f"index entry {tname} -> {f}: not in that shard")
            continue
        tensors[tname] = t
    for f, s in shards.items():
        for tname in s.tensors:
            if wmap.get(tname) != f:
                problems.append(f"{f}: {tname} is not mapped to this shard by the index ({wmap.get(tname)})")
    total = sum(t.nbytes for t in tensors.values())
    if index.get("metadata", {}).get("total_size") != total:
        problems.append(f"index total_size {index.get('metadata', {}).get('total_size')} != sum of tensor bytes {total}")
    return Checkpoint(path, config, config_raw, quant, quant_raw, index, index_raw, shards, tensors, problems)


# ----------------------------------------------------------------------------- the subset rule

def classify(name: str, include_visual: bool) -> str:
    m = _LAYER.match(name)
    if m:
        return "keep" if int(m.group(1)) < NUM_LAYERS else "drop:layers"
    if name.startswith(GLOBAL_KEEP):
        return "keep"
    if name.startswith(VISUAL):
        return "keep" if include_visual else "drop:visual"
    raise Fail(f"unexpected tensor {name!r}: not a decoder layer, embedding, final norm, lm_head, mtp or visual")


def group_of(name: str) -> str:
    m = _LAYER.match(name)
    if m:
        return f"layers.{m.group(1)}"
    for p in GLOBAL_KEEP + (VISUAL,):
        if name.startswith(p):
            return p.rstrip(".").replace("model.language_model.", "").replace("model.", "")
    return "other"


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def mini_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    tc = text_config(out)
    lt = tc["layer_types"]
    if len(lt) != tc["num_hidden_layers"]:
        raise Fail(f"layer_types has {len(lt)} entries, num_hidden_layers is {tc['num_hidden_layers']}")
    if tc["num_hidden_layers"] < NUM_LAYERS:
        raise Fail(f"source has only {tc['num_hidden_layers']} layers")
    tc["num_hidden_layers"] = NUM_LAYERS
    tc["layer_types"] = lt[:NUM_LAYERS]
    if "full_attention" not in tc["layer_types"]:
        raise Fail(f"layer_types[:{NUM_LAYERS}] has no full_attention layer: {tc['layer_types']}")
    return out


def mini_quant(qc: dict, include_visual: bool) -> dict:
    out_ts = {}
    for mod, entry in qc["tensor_storage"].items():
        states = {classify(n, include_visual) == "keep" for n in entry["stored_tensors"]}
        if len(states) != 1:
            raise Fail(f"tensor_storage module {mod} is split by the subset rule")
        if states == {True}:
            out_ts[mod] = entry
    out = dict(qc)
    out["tensor_storage"] = out_ts
    return out


def json_style(obj, raw: bytes) -> tuple[int, str]:
    """(indent, trailer) under which json.dumps reproduces the source file, so diffs stay minimal."""
    for indent in (2, 4, 1, 3):
        text = json.dumps(obj, indent=indent, ensure_ascii=False)
        for trailer in ("", "\n"):
            if (text + trailer).encode() == raw:
                return indent, trailer
    return 2, "\n"


def dumps_like(obj, like_obj, like_raw: bytes) -> bytes:
    indent, trailer = json_style(like_obj, like_raw)
    return (json.dumps(obj, indent=indent, ensure_ascii=False) + trailer).encode()


@dataclass
class DestShard:
    file: str
    source: str
    tensors: list           # [Tensor] (source records) in destination data order
    header: bytes           # JSON header, space-padded to a multiple of 8 bytes

    @property
    def data_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def size(self) -> int:
        return 8 + len(self.header) + self.data_bytes


@dataclass
class Plan:
    kept: list
    dropped: Counter
    shards: list
    weight_map: dict
    config: dict
    quant: dict


def make_plan(src: Checkpoint, include_visual: bool) -> Plan:
    kept, dropped = [], Counter()
    for t in src.tensors.values():
        c = classify(t.name, include_visual)
        if c == "keep":
            kept.append(t)
        else:
            dropped[c] += 1
    by_src = defaultdict(list)
    for t in kept:
        by_src[t.file].append(t)
    n = len(by_src)
    shards, where = [], {}
    for j, sf in enumerate(sorted(by_src), 1):
        # Widest dtype first, then source order: with an 8-aligned data start every tensor begins
        # on a multiple of its element size (tensor sizes are multiples of their element size).
        ts = sorted(by_src[sf], key=lambda t: (-DTYPE_SIZE[t.dtype], t.begin))
        name = f"model-{j:05d}-of-{n:05d}.safetensors"
        header, off = {}, 0
        meta = src.shards[sf].metadata
        if meta is not None:
            header["__metadata__"] = meta
        for t in ts:
            header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
            off += t.nbytes
            where[t.name] = name
        raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
        raw += b" " * (-len(raw) % 8)
        shards.append(DestShard(name, sf, ts, raw))
    weight_map = {t.name: where[t.name] for t in kept}   # source index order
    return Plan(kept, dropped, shards, weight_map, mini_config(src.config), mini_quant(src.quant, include_visual))


def expected_count_from_config(cfg: dict, include_visual: bool, n_visual_src: int) -> tuple[int, str]:
    """Tensor count of the subset from the architecture alone (cross-check of the index-derived count)."""
    tc = text_config(cfg)
    if tc.get("mtp_use_dedicated_embeddings"):
        raise Fail("mtp_use_dedicated_embeddings=true is not modeled")
    e = tc["num_experts"]
    common = 2 + 1 + 1 + 3 * 4 + e * 3 * 4          # 2 norms, router, shared gate, shared expert, experts
    lin = common + 3 * 4 + 6                         # in_proj_qkv/z, out_proj + in_proj_a/b, conv1d, A_log, dt_bias, norm
    full = common + 4 * 4 + 2                        # q/k/v/o + q_norm, k_norm
    types = tc["layer_types"][:NUM_LAYERS]
    layers = sum(lin if t == "linear_attention" else full for t in types)
    mtp = 4 + 3 + tc["mtp_num_hidden_layers"] * full  # fc, 3 norms, MTP decoder layer(s)
    total = layers + 1 + 1 + 4 + mtp + (n_visual_src if include_visual else 0)
    nl = Counter(types)
    formula = (f"{nl['linear_attention']} x {lin} (GDN layer) + {nl['full_attention']} x {full} (attention layer)"
               f" + 1 embed + 1 norm + 4 lm_head + {mtp} mtp (= 4 fc + 3 norms + {tc['mtp_num_hidden_layers']} x {full})"
               + (f" + {n_visual_src} visual" if include_visual else "") + f" = {total}")
    return total, formula


# ----------------------------------------------------------------------------- byte IO

def _pread_exact(fd: int, n: int, off: int) -> bytes:
    buf = os.pread(fd, n, off)
    if len(buf) != n:
        parts = [buf]
        got = len(buf)
        while got < n:
            b = os.pread(fd, n - got, off + got)
            if not b:
                raise Fail(f"short read at offset {off + got}")
            parts.append(b)
            got += len(b)
        buf = b"".join(parts)
    return buf


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def sha256_range(fd: int, begin: int, nbytes: int) -> str:
    h = hashlib.sha256()
    pos, end = begin, begin + nbytes
    while pos < end:
        n = min(CHUNK, end - pos)
        h.update(_pread_exact(fd, n, pos))
        pos += n
    return h.hexdigest()


def sha256_file(path: str) -> str:
    fd = os.open(path, os.O_RDONLY)
    try:
        return sha256_range(fd, 0, os.fstat(fd).st_size)
    finally:
        os.close(fd)


def copy_range(src_fd: int, begin: int, nbytes: int, dst_fd: int) -> str:
    h = hashlib.sha256()
    pos, end = begin, begin + nbytes
    while pos < end:
        n = min(CHUNK, end - pos)
        buf = _pread_exact(src_fd, n, pos)
        h.update(buf)
        _write_all(dst_fd, buf)
        pos += n
    return h.hexdigest()


def write_file(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def gib(n: int) -> str:
    return f"{n / 2**30:.3f} GiB"


# ----------------------------------------------------------------------------- create

def cmd_create(args) -> int:
    src = load_checkpoint(args.src)
    if src.problems:
        raise Fail("source checkpoint is inconsistent:\n  " + "\n  ".join(src.problems[:20]))
    dst = os.path.abspath(args.dst)
    part = dst + ".partial"
    for p in (dst, part):
        if os.path.lexists(p):
            raise Fail(f"{p} exists; refusing to overwrite (remove it yourself if it is stale)")
    parent = os.path.dirname(dst)
    if not os.path.isdir(parent):
        raise Fail(f"parent directory {parent} does not exist")
    if os.path.realpath(dst).startswith(os.path.realpath(src.path) + os.sep):
        raise Fail("destination is inside the source directory")

    plan = make_plan(src, args.include_visual)
    aux = [(s, d) for s, d, req in AUX_FILES if os.path.isfile(os.path.join(src.path, s))]
    missing = [s for s, _, req in AUX_FILES if req and not os.path.isfile(os.path.join(src.path, s))]
    if missing:
        raise Fail(f"required files missing in the source: {missing}")
    if not any(s in CHAT_TEMPLATES for s, _ in aux):
        print("warning: the source has no chat template file", file=sys.stderr)
    need = sum(s.size for s in plan.shards) + sum(os.path.getsize(os.path.join(src.path, s)) for s, _ in aux)
    free = shutil.disk_usage(parent).free
    margin = 1 << 30
    groups = Counter(group_of(t.name) for t in plan.kept)
    print(f"source        {src.path}")
    print(f"destination   {dst}")
    print(f"kept tensors  {len(plan.kept)} ({gib(sum(t.nbytes for t in plan.kept))}); by group: {dict(sorted(groups.items()))}")
    print(f"dropped       {dict(plan.dropped)}")
    for s in plan.shards:
        print(f"shard         {s.file}  <- {s.source}: {len(s.tensors)} tensors, {s.size:,} B")
    print(f"aux files     {[d for _, d in aux]}")
    print(f"space         need {need:,} B ({gib(need)}) + {gib(margin)} margin; free on {parent}: {free:,} B ({gib(free)})")
    if need + margin > free:
        raise Fail("not enough free space")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    os.mkdir(part)
    t0 = time.monotonic()
    tensor_records = {}
    src_fds = {}
    try:
        for s in plan.shards:
            if s.source not in src_fds:
                src_fds[s.source] = os.open(os.path.join(src.path, s.source), os.O_RDONLY)
            sfd = src_fds[s.source]
            dfd = os.open(os.path.join(part, s.file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            ts = time.monotonic()
            try:
                _write_all(dfd, struct.pack("<Q", len(s.header)) + s.header)
                for t in s.tensors:
                    digest = copy_range(sfd, t.begin, t.nbytes, dfd)
                    tensor_records[t.name] = {"sha256": digest, "file": s.file,
                                              "source_file": t.file, "source_range": [t.begin, t.begin + t.nbytes]}
                os.fsync(dfd)
            finally:
                os.close(dfd)
            got = os.path.getsize(os.path.join(part, s.file))
            if got != s.size:
                raise Fail(f"{s.file}: wrote {got} B, planned {s.size} B")
            dt = time.monotonic() - ts
            print(f"wrote         {s.file}: {got:,} B in {dt:.1f} s ({got / 2**20 / max(dt, 1e-9):.0f} MiB/s)")
    finally:
        for fd in src_fds.values():
            os.close(fd)

    total = sum(t.nbytes for t in plan.kept)
    index = {"metadata": {"total_size": total}, "weight_map": plan.weight_map}
    write_file(os.path.join(part, "model.safetensors.index.json"), dumps_like(index, src.index, src.index_raw))
    write_file(os.path.join(part, "config.json"), dumps_like(plan.config, src.config, src.config_raw))
    write_file(os.path.join(part, "quantization_config.json"), dumps_like(plan.quant, src.quant, src.quant_raw))
    copied = {}
    for s, d in aux:
        with open(os.path.join(src.path, s), "rb") as f:
            data = f.read()
        write_file(os.path.join(part, d), data)
        copied[d] = {"source": s, "sha256": hashlib.sha256(data).hexdigest()}
    prov = {
        "tool": "make_ornith_mini.py", "tool_version": TOOL_VERSION,
        "source": {"path": src.path,
                   "config_sha256": hashlib.sha256(src.config_raw).hexdigest(),
                   "quantization_config_sha256": hashlib.sha256(src.quant_raw).hexdigest(),
                   "index_sha256": hashlib.sha256(src.index_raw).hexdigest()},
        "num_layers": NUM_LAYERS, "include_visual": bool(args.include_visual),
        "serve_with": [] if args.include_visual else ["--language-model-only"],
        "kept": {"tensors": len(plan.kept), "bytes": total, "by_group": dict(sorted(groups.items()))},
        "dropped": dict(plan.dropped),
        "shards": {s.file: {"source": s.source, "tensors": len(s.tensors), "size": s.size} for s in plan.shards},
        "files": copied,
        "tensors": tensor_records,
    }
    write_file(os.path.join(part, PROVENANCE), (json.dumps(prov, indent=1) + "\n").encode())
    dfd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    os.rename(part, dst)
    size = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"done          {dst}: {len(os.listdir(dst))} files, {size:,} B ({gib(size)}) in {time.monotonic() - t0:.1f} s")
    print(f"next          {os.path.basename(sys.argv[0])} verify --src {src.path} --dst {dst}")
    return 0


# ----------------------------------------------------------------------------- verify

class Report:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append(("PASS" if ok else "FAIL", name, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)
        return ok

    def info(self, name: str, detail: str) -> None:
        self.rows.append(("INFO", name, detail))
        print(f"INFO  {name}: {detail}", flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r[0] == "FAIL")


def json_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                yield p
            else:
                yield from json_diff(a[k], b[k], p)
    elif a != b or type(a) is not type(b):
        yield path


def exl3_family(module: str) -> str:
    m = _LAYER.match(module + ".") or _MTP_LAYER.match(module + ".")
    rest = module[m.end():] if m else module
    return _EXPERT.sub("experts.N.", rest)


def expected_dims(cfg: dict) -> dict:
    tc = text_config(cfg)
    h = tc["hidden_size"]
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    q_out = nh * hd * (2 if tc.get("attn_output_gate") else 1)
    kh, dk = tc["linear_num_key_heads"], tc["linear_key_head_dim"]
    vh, dv = tc["linear_num_value_heads"], tc["linear_value_head_dim"]
    im, ish = tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"]
    return {
        "linear_attn.in_proj_qkv": (h, 2 * kh * dk + vh * dv), "linear_attn.in_proj_z": (h, vh * dv),
        "linear_attn.out_proj": (vh * dv, h),
        "self_attn.q_proj": (h, q_out), "self_attn.k_proj": (h, nkv * hd), "self_attn.v_proj": (h, nkv * hd),
        "self_attn.o_proj": (nh * hd, h),
        "mlp.shared_expert.gate_proj": (h, ish), "mlp.shared_expert.up_proj": (h, ish),
        "mlp.shared_expert.down_proj": (ish, h),
        "mlp.experts.N.gate_proj": (h, im), "mlp.experts.N.up_proj": (h, im), "mlp.experts.N.down_proj": (im, h),
        "lm_head": (h, tc["vocab_size"]), "mtp.fc": (2 * h, h),
    }


def all_zero_f16(data: bytes) -> bool:
    """True if every F16 value is +0 or -0."""
    mask = int.from_bytes(b"\xff\x7f" * (len(data) // 2), "little")
    return int.from_bytes(data, "little") & mask == 0


class Reader:
    def __init__(self, ck: Checkpoint):
        self.ck = ck
        self.fds = {f: os.open(os.path.join(ck.path, f), os.O_RDONLY) for f in ck.shards}

    def read(self, t: Tensor) -> bytes:
        return _pread_exact(self.fds[t.file], t.nbytes, t.begin)

    def sha(self, t: Tensor) -> str:
        return sha256_range(self.fds[t.file], t.begin, t.nbytes)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def cmd_verify(args) -> int:
    r = Report()
    src = load_checkpoint(args.src)
    if src.problems:
        raise Fail("source checkpoint is inconsistent:\n  " + "\n  ".join(src.problems[:20]))
    prov_path = os.path.join(args.dst, PROVENANCE)
    prov = _read_json(prov_path)[0] if os.path.isfile(prov_path) else None
    include_visual = bool(prov and prov.get("include_visual")) if args.include_visual is None else args.include_visual
    plan = make_plan(src, include_visual)
    dst = load_checkpoint(args.dst)
    print(f"source {src.path}\ndest   {dst.path}\ninclude_visual={include_visual}")

    # 1. structure of the new shards and index
    r.check("dest shards and index are consistent", not dst.problems,
            f"{len(dst.shards)} shards, {len(dst.tensors)} index entries" if not dst.problems
            else "; ".join(dst.problems[:10]))
    misaligned = [t.name for t in dst.tensors.values() if t.begin % DTYPE_SIZE[t.dtype]]
    starts = {f: s.data_start % 8 for f, s in dst.shards.items()}
    r.check("every dest tensor starts on a multiple of its element size (8-aligned data start)",
            not misaligned and not any(starts.values()), f"misaligned {len(misaligned)}, data start mod 8 {starts}")

    # 2. tensor set and count
    n_vis = sum(1 for n in src.tensors if n.startswith(VISUAL))
    want = {t.name for t in plan.kept}
    have = set(dst.tensors)
    groups_src = Counter(group_of(n) for n in src.tensors)
    groups = Counter(group_of(n) for n in want)
    keep_groups = [g for g in groups_src if g in groups]
    idx_formula = " + ".join(f"{groups_src[g]} {g}" for g in sorted(keep_groups)) + f" = {len(want)}"
    r.check("tensor set == source index entries selected by the subset rule", have == want,
            f"missing {len(want - have)}, unexpected {len(have - want)}"
            + (f", e.g. {sorted(want - have)[:3]} {sorted(have - want)[:3]}" if have != want else ""))
    r.check("tensor count from the source index", len(have) == len(want), f"{len(have)} == {idx_formula}")
    n_arch, arch_formula = expected_count_from_config(src.config, include_visual, n_vis)
    r.check("tensor count from the architecture (config.json)", n_arch == len(want), arch_formula)
    r.check("dest index weight_map order and shard assignment as planned",
            list(dst.index["weight_map"].items()) == list(plan.weight_map.items()))

    # 3. dtype / shape / size per tensor
    bad = [n for n in want & have if (dst.tensors[n].dtype, dst.tensors[n].shape, dst.tensors[n].nbytes)
           != (src.tensors[n].dtype, src.tensors[n].shape, src.tensors[n].nbytes)]
    r.check("dtype, shape and byte size equal the source for every tensor", not bad, f"{len(bad)} differ {bad[:3]}")
    size = sum(dst.tensors[n].nbytes for n in have)
    r.check("index total_size", dst.index["metadata"]["total_size"] == size == sum(t.nbytes for t in plan.kept),
            f"{size:,} B ({gib(size)})")

    # 4. sha256 per tensor: dest bytes vs the source byte range (and vs the record written by create)
    rs, rd = Reader(src), Reader(dst)
    t0 = time.monotonic()
    names = sorted(want & have, key=lambda n: -src.tensors[n].nbytes)

    def job(n):
        return n, rs.sha(src.tensors[n]), rd.sha(dst.tensors[n])

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(job, names))
    dt = time.monotonic() - t0
    mism = [n for n, a, b in results if a != b]
    r.check("sha256(dest tensor) == sha256(source byte range), every tensor", not mism and len(results) == len(want),
            f"{len(results)} tensors, {2 * size / 2**30:.2f} GiB hashed in {dt:.1f} s, {len(mism)} mismatch {mism[:3]}")
    if prov is not None:
        recs = prov.get("tensors", {})
        pm = [n for n, a, _ in results if recs.get(n, {}).get("sha256") != a]
        r.check(f"sha256 == the digests {PROVENANCE} recorded at copy time", not pm and len(recs) == len(want),
                f"{len(recs)} records, {len(pm)} mismatch")
        rng = [n for n in want if tuple(recs.get(n, {}).get("source_range", ())) !=
               (src.tensors[n].begin, src.tensors[n].begin + src.tensors[n].nbytes)]
        r.check("recorded source byte ranges equal the source headers", not rng, f"{len(rng)} differ")
    else:
        r.info(PROVENANCE, "absent; skipped the copy-time digest comparison")

    # 5. config.json
    diffs = sorted(set(json_diff(src.config, dst.config)))
    tc_path = "text_config." if "text_config" in src.config else ""
    allowed = {f"{tc_path}num_hidden_layers", f"{tc_path}layer_types"}
    stc, dtc = text_config(src.config), text_config(dst.config)
    r.check("config.json differs from the source only in num_hidden_layers and layer_types",
            set(diffs) <= allowed, f"differing paths {diffs}")
    r.check("config.json num_hidden_layers / layer_types",
            dtc.get("num_hidden_layers") == NUM_LAYERS and dtc.get("layer_types") == stc["layer_types"][:NUM_LAYERS],
            f"{stc['num_hidden_layers']} -> {dtc.get('num_hidden_layers')}, layer_types {dtc.get('layer_types')}")
    for key in ("quantization_config", "rope_parameters", "mtp_num_hidden_layers", "num_experts",
                "full_attention_interval", "mamba_ssm_dtype"):
        a = src.config.get(key, stc.get(key))
        b = dst.config.get(key, dtc.get(key))
        r.check(f"config.json {key} unchanged", a == b and a is not None, json.dumps(b)[:120])
    r.check("config.json bytes == source re-serialized with only those two fields changed",
            open(os.path.join(dst.path, "config.json"), "rb").read() == dumps_like(plan.config, src.config, src.config_raw))

    # 6. quantization_config.json
    qd, qs = dst.quant, src.quant
    r.check("quantization_config.json top-level fields unchanged",
            {k: v for k, v in qd.items() if k != "tensor_storage"} == {k: v for k, v in qs.items() if k != "tensor_storage"},
            json.dumps({k: v for k, v in qd.items() if k != "tensor_storage"}))
    r.check("quantization_config.json tensor_storage == source entries of the kept modules (same order, same content)",
            list(qd["tensor_storage"].items()) == list(plan.quant["tensor_storage"].items()),
            f"{len(qd['tensor_storage'])} of {len(qs['tensor_storage'])} modules")
    st_bad, st_names = [], set()
    for mod, e in qd["tensor_storage"].items():
        for n, info in e["stored_tensors"].items():
            st_names.add(n)
            t = dst.tensors.get(n)
            if t is None or list(t.shape) != info["shape"] or t.nbytes != info["n_bytes"] \
                    or info["dtype"].replace("torch.", "") != {"F16": "float16", "BF16": "bfloat16", "I16": "int16",
                                                               "I32": "int32", "F32": "float32"}.get(t.dtype):
                st_bad.append(n)
    r.check("every tensor_storage tensor is in the dest with that dtype, shape and n_bytes", not st_bad,
            f"{len(st_names)} tensors, {len(st_bad)} bad {st_bad[:3]}")
    outside = Counter("mtp.*" if n.startswith("mtp.") else re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
                      for n in have - st_names)
    r.info("dest tensors not in tensor_storage (the source omits these too)", json.dumps(dict(sorted(outside.items()))))
    r.check("tensor_storage omissions are only mtp.*, A_log, conv1d, dt_bias (and visual)",
            set(outside) <= {"mtp.*", "model.language_model.layers.N.linear_attn.A_log",
                             "model.language_model.layers.N.linear_attn.conv1d.weight",
                             "model.language_model.layers.N.linear_attn.dt_bias"}
            | {k for k in outside if k.startswith(VISUAL)})

    # 7. aux files
    aux_bad, aux_seen = [], []
    for s, d, req in AUX_FILES:
        sp, dp = os.path.join(src.path, s), os.path.join(dst.path, d)
        if not os.path.isfile(sp):
            continue
        aux_seen.append(d)
        if not os.path.isfile(dp) or sha256_file(sp) != sha256_file(dp):
            aux_bad.append(d)
    r.check("tokenizer / chat template / generation / processor files byte-identical to the source",
            not aux_bad, f"{aux_seen}; bad {aux_bad}")
    r.info("dest directory", json.dumps(sorted(os.listdir(dst.path))))

    # 8. EXL3 manifest: mul1, K in {4,5,6}, widths multiples of 128, dims from config, K by family
    dims = expected_dims(src.config)
    mods = defaultdict(dict)
    for n in have:
        mod, _, suf = n.rpartition(".")
        mods[mod][suf] = dst.tensors[n]
    exl3 = {m: s for m, s in mods.items() if "trellis" in s}
    errs = []
    kfam = Counter()
    for m, s in exl3.items():
        if set(s) != EXL3_SUFFIXES:
            errs.append(f"{m}: tensors {sorted(s)}")
            continue
        tr, suh, svh, mk = s["trellis"], s["suh"], s["svh"], s["mul1"]
        if tr.dtype != "I16" or len(tr.shape) != 3 or tr.shape[2] % 16:
            errs.append(f"{m}: trellis {tr.dtype} {tr.shape}")
            continue
        k_in, n_out, kk = 16 * tr.shape[0], 16 * tr.shape[1], tr.shape[2] // 16
        if kk not in ALLOWED_K:
            errs.append(f"{m}: K={kk}")
        if k_in % BLOCK or n_out % BLOCK:
            errs.append(f"{m}: widths {k_in}x{n_out}")
        if (suh.dtype, suh.shape, svh.dtype, svh.shape) != ("F16", (k_in,), "F16", (n_out,)):
            errs.append(f"{m}: suh {suh.dtype}{suh.shape} svh {svh.dtype}{svh.shape}")
        if mk.dtype != "I32" or mk.shape not in ((), (1,)):
            errs.append(f"{m}: mul1 {mk.dtype} {mk.shape}")
        elif struct.unpack("<I", rd.read(mk))[0] != MUL1_MARKER:
            errs.append(f"{m}: mul1 marker {struct.unpack('<I', rd.read(mk))[0]:#x}")
        fam = exl3_family(m)
        if dims.get(fam) != (k_in, n_out):
            errs.append(f"{m}: {k_in}x{n_out}, config says {dims.get(fam)}")
        role = "mtp" if m.startswith("mtp.") else "lm_head" if m == "lm_head" else \
            "experts" if ".experts." in m else "dense"
        if RECIPE_K[role] != kk:
            errs.append(f"{m}: K={kk}, recipe {role} K={RECIPE_K[role]}")
        ent = qd["tensor_storage"].get(m)
        if ent is not None and (ent.get("quant_format") != "exl3" or ent.get("bits_per_weight") != kk
                                or ent.get("mul1_multiplier") != MUL1_MARKER):
            errs.append(f"{m}: tensor_storage {ent.get('quant_format')} K={ent.get('bits_per_weight')} "
                        f"mul1={ent.get('mul1_multiplier')}")
        kfam[(role, kk)] += 1
    others_with_scales = [m for m, s in mods.items() if "trellis" not in s and ({"suh", "svh", "mcg", "mul1"} & set(s))]
    r.check("every EXL3 matrix: exactly trellis/suh/svh/mul1, mul1 marker 0x83DCD12D, K in {4,5,6}, "
            "in/out multiples of 128 and equal to config widths, K as published per family",
            not errs and not others_with_scales,
            f"{len(exl3)} matrices; K by role {dict(sorted(kfam.items()))}"
            + (f"; {len(errs)} errors {errs[:5]}" if errs else "")
            + (f"; scale tensors without trellis {others_with_scales[:3]}" if others_with_scales else ""))

    dense = [m for m in exl3 if not m.startswith("mtp.") and ".experts." not in m]
    mtp_dense = [m for m in exl3 if m.startswith("mtp.") and ".experts." not in m]
    experts = [m for m in exl3 if not m.startswith("mtp.") and ".experts." in m]
    mtp_experts = [m for m in exl3 if m.startswith("mtp.") and ".experts." in m]
    e = text_config(src.config)["num_experts"]
    r.check("dense EXL3 (DESIGN 7: 26 matrices x 4 = 104 tensors)", len(dense) == 26,
            f"{len(dense)} matrices = {Counter(exl3_family(m) for m in dense).most_common()} -> {4 * len(dense)} tensors")
    r.check("MTP dense EXL3 (DESIGN 7: +32 tensors)", len(mtp_dense) == 8,
            f"{len(mtp_dense)} matrices {sorted(exl3_family(m) for m in mtp_dense)} -> {4 * len(mtp_dense)} tensors")
    r.check(f"routed-expert EXL3 ({NUM_LAYERS} layers x {e} x 3)", len(experts) == NUM_LAYERS * e * 3,
            f"{len(experts)} matrices -> {4 * len(experts)} tensors")
    r.check(f"MTP routed-expert EXL3 ({e} x 3)", len(mtp_experts) == e * 3,
            f"{len(mtp_experts)} matrices -> {4 * len(mtp_experts)} tensors")

    # 9. exact-zero probes: dead experts (gate, up and down svh all zero) in source and dest
    dead_src, dead_dst = set(), set()
    for li in range(NUM_LAYERS):
        for x in range(e):
            base = f"model.language_model.layers.{li}.mlp.experts.{x}"
            svh = [f"{base}.{p}.svh" for p in ("gate_proj", "up_proj", "down_proj")]
            if all(all_zero_f16(rs.read(src.tensors[n])) for n in svh):
                dead_src.add((li, x))
            if all(n in dst.tensors and all_zero_f16(rd.read(dst.tensors[n])) for n in svh):
                dead_dst.add((li, x))
    probes = {(li, x) for li, xs in DEAD_EXPERT_PROBES.items() for x in xs}
    r.check(f"probe experts {sorted(probes)} (layer, expert): gate/up/down svh exact zeros in the source and in the dest",
            probes <= dead_src and probes <= dead_dst)
    r.check(f"they are the only all-zero-svh experts in layers 0-{NUM_LAYERS - 1}",
            dead_src == dead_dst == probes, f"source {sorted(dead_src)}, dest {sorted(dead_dst)}")
    detail = []
    for li, x in sorted(probes & dead_dst):
        base = f"model.language_model.layers.{li}.mlp.experts.{x}"
        nz = {p: any(rd.read(dst.tensors[f"{base}.{p}.trellis"])) for p in ("gate_proj", "up_proj", "down_proj")}
        suh = {p: not all_zero_f16(rd.read(dst.tensors[f"{base}.{p}.suh"])) for p in ("gate_proj", "up_proj", "down_proj")}
        detail.append(f"{x}: trellis nonzero {sum(nz.values())}/3, suh nonzero {sum(suh.values())}/3")
    r.info("probe experts (output is 0 because svh is 0, not because the trellis is)", "; ".join(detail))
    rs.close()
    rd.close()

    # 10. second opinion: trellis_core's manifest reader (pure Python, same repo), if importable
    core = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "src")
    try:
        sys.path.insert(0, core)
        from trellis_core.format import Codebook, load_manifest
        man = load_manifest(dst.path)
        ks = Counter(m.bits.value for m in man.matrices.values())
        cbs = {m.codebook for m in man.matrices.values()}
        r.check("trellis_core.load_manifest(dest): matrix count, codebook mul1, K set",
                len(man.matrices) == len(exl3) and cbs == {Codebook.MUL1} and set(ks) <= set(ALLOWED_K)
                and len(man.unquantized) == len(have) - 4 * len(exl3),
                f"{len(man.matrices)} matrices, K {dict(sorted(ks.items()))}, {len(man.unquantized)} unquantized, "
                f"warnings {man.warnings}")
    except ImportError as ex:
        r.info("trellis_core cross-check", f"skipped ({ex})")
    finally:
        if sys.path and sys.path[0] == core:
            sys.path.pop(0)

    total = sum(os.path.getsize(os.path.join(dst.path, f)) for f in os.listdir(dst.path))
    r.info("dest size", f"{total:,} B ({gib(total)}) in {len(os.listdir(dst.path))} files")
    print(f"\n{'OK' if not r.failed else 'FAILED'}: {sum(1 for x in r.rows if x[0] == 'PASS')} passed, {r.failed} failed")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([{"status": s, "check": c, "detail": d} for s, c, d in r.rows], f, indent=1)
    return 1 if r.failed else 0


# ----------------------------------------------------------------------------- check-config (vLLM image)

def cmd_check_config(args) -> int:
    """transformers AutoConfig + vLLM ModelConfig on the mini; no weights loaded, no device opened."""
    r = Report()
    os.environ.setdefault("EXL3_VLLM_PATCHES", "0")
    from transformers import AutoConfig

    dst = AutoConfig.from_pretrained(args.dst)
    dtc = dst.get_text_config()
    r.check("transformers AutoConfig parses the mini", True,
            f"{type(dst).__name__} model_type={dst.model_type}, text {type(dtc).__name__}")
    r.check("text config: 4 layers, layer_types[:4], MTP, experts",
            dtc.num_hidden_layers == NUM_LAYERS and list(dtc.layer_types) == ["linear_attention"] * 3 + ["full_attention"]
            and dtc.mtp_num_hidden_layers == 1 and dtc.num_experts == 256,
            f"num_hidden_layers={dtc.num_hidden_layers} layer_types={list(dtc.layer_types)} "
            f"mtp_num_hidden_layers={dtc.mtp_num_hidden_layers} num_experts={dtc.num_experts}")
    if args.src:
        src = AutoConfig.from_pretrained(args.src)
        diffs = sorted(set(json_diff(src.to_dict(), dst.to_dict())))
        r.check("AutoConfig(src).to_dict() vs AutoConfig(mini).to_dict(): only num_hidden_layers / layer_types differ",
                set(diffs) <= {"text_config.num_hidden_layers", "text_config.layer_types", "_name_or_path",
                               "text_config._name_or_path", "vision_config._name_or_path"}, f"{diffs}")

    import safetensors
    import torch
    from safetensors import safe_open

    wmap = _read_json(os.path.join(args.dst, "model.safetensors.index.json"))[0]["weight_map"]
    bad, n_ok = [], 0
    for f in sorted(set(wmap.values())):
        with safe_open(os.path.join(args.dst, f), framework="pt") as st:
            keys = set(st.keys())
            if keys != {n for n, g in wmap.items() if g == f}:
                bad.append(f)
            for n in keys:
                sl = st.get_slice(n)
                sl.get_shape(), sl.get_dtype()
            n_ok += len(keys)
    r.check("safetensors.safe_open parses every shard, keys == index", not bad,
            f"safetensors {safetensors.__version__}, {n_ok} tensors, bad shards {bad}")
    if args.src:
        swmap = _read_json(os.path.join(args.src, "model.safetensors.index.json"))[0]["weight_map"]
        sample = ["lm_head.mul1", "lm_head.svh", "model.language_model.norm.weight", "mtp.fc.suh",
                  "model.language_model.layers.0.linear_attn.A_log",
                  "model.language_model.layers.3.mlp.experts.72.gate_proj.svh"]
        diff = []
        for n in sample:
            with safe_open(os.path.join(args.dst, wmap[n]), framework="pt") as a, \
                    safe_open(os.path.join(args.src, swmap[n]), framework="pt") as b:
                ta, tb = a.get_tensor(n), b.get_tensor(n)
                if ta.dtype != tb.dtype or ta.shape != tb.shape or not torch.equal(ta, tb):
                    diff.append(n)
        r.check("safetensors get_tensor on sample tensors equals the source (torch.equal)", not diff,
                f"{sample}; differ {diff}")

    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import ParallelConfig
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    plugin_err = None
    try:
        from exl3xpu import vllm_plugin
        vllm_plugin.register()          # registers 'exl3' (and installs the graph guard, inert on CPU)
    except Exception as ex:   # report, then fall back to a stub so the config parse is still checked
        plugin_err = f"{type(ex).__name__}: {ex}"
        vllm_plugin = None
        from vllm.model_executor.layers.quantization import register_quantization_config
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

        @register_quantization_config("exl3")
        class _StubExl3(QuantizationConfig):
            def get_name(self): return "exl3"
            def get_supported_act_dtypes(self): return []
            @classmethod
            def get_min_capability(cls): return 0
            @staticmethod
            def get_config_filenames(): return []
            @classmethod
            def from_config(cls, config): return cls()
            def get_quant_method(self, layer, prefix): return None
    r.check("exl3xpu.vllm_plugin imports and registers 'exl3'", plugin_err is None and "exl3" in QUANTIZATION_METHODS,
            plugin_err or f"vLLM {vllm.__version__}")

    def model_config(path):
        return EngineArgs(model=path, tokenizer=path, dtype="bfloat16", quantization="exl3",
                          language_model_only=True, trust_remote_code=False).create_model_config()

    pc = ParallelConfig()
    mc = model_config(args.dst)
    n_attn = mc.get_num_layers_by_block_type(pc, "attention")
    n_lin = mc.get_num_layers_by_block_type(pc, "linear_attention")
    r.check("vLLM ModelConfig (EngineArgs.create_model_config, --language-model-only, --quantization exl3)",
            mc.architecture == "Qwen3_5MoeForConditionalGeneration" and mc.quantization == "exl3"
            and mc.hf_text_config.num_hidden_layers == NUM_LAYERS and n_attn == 1 and n_lin == 3 and mc.is_hybrid
            and mc.multimodal_config is not None and mc.multimodal_config.language_model_only,
            f"architecture={mc.architecture} quantization={mc.quantization} layers={mc.get_num_layers(pc)} "
            f"attention={n_attn} linear_attention={n_lin} hybrid={mc.is_hybrid} kv_heads={mc.get_total_num_kv_heads()} "
            f"head_size={mc.get_head_size()} hidden={mc.get_hidden_size()} vocab={mc.get_vocab_size()} "
            f"max_model_len={mc.max_model_len} dtype={mc.dtype} "
            f"limit_mm(image)={mc.multimodal_config.get_limit_per_prompt('image') if mc.multimodal_config else None}")
    if args.src:
        ms = model_config(args.src)
        same = {k: (getattr(ms, k), getattr(mc, k)) for k in ("architecture", "quantization", "max_model_len", "dtype")}
        same["kv_heads"] = (ms.get_total_num_kv_heads(), mc.get_total_num_kv_heads())
        same["head_size"] = (ms.get_head_size(), mc.get_head_size())
        same["vocab"] = (ms.get_vocab_size(), mc.get_vocab_size())
        r.check("vLLM ModelConfig of the source agrees on everything but the layer count",
                all(a == b for a, b in same.values()) and ms.get_num_layers(pc) == 48,
                f"source layers={ms.get_num_layers(pc)} attention={ms.get_num_layers_by_block_type(pc, 'attention')}; "
                + ", ".join(f"{k}={a}" for k, (a, b) in same.items()))
    if vllm_plugin is not None:
        q = vllm_plugin.Exl3Config.from_config(mc.hf_config.quantization_config)
        q.maybe_update_config(args.dst)
        ks = Counter(q.storage.values())
        n_mtp = sum(1 for k in q.storage if k.startswith("mtp."))
        r.check("Exl3Config.maybe_update_config(mini): EXL3 module map",
                len(q.storage) == 26 + NUM_LAYERS * 256 * 3 + 776 and n_mtp == 776,
                f"{len(q.storage)} modules ({n_mtp} MTP), K histogram {dict(sorted(ks.items()))}")
    try:
        from vllm.config import SpeculativeConfig
        sc = SpeculativeConfig(method="mtp", num_speculative_tokens=3, target_model_config=mc, target_parallel_config=pc)
        dm = sc.draft_model_config
        r.info("SpeculativeConfig(method=mtp, k=3) on the mini",
               f"draft architecture={dm.architecture} model_type={dm.hf_config.model_type} "
               f"n_predict={getattr(dm.hf_config, 'n_predict', None)} quantization={dm.quantization}")
    except Exception as ex:
        r.info("SpeculativeConfig(method=mtp, k=3) on the mini", f"not constructed on CPU: {type(ex).__name__}: {ex}")
    print(f"\n{'OK' if not r.failed else 'FAILED'}: {sum(1 for x in r.rows if x[0] == 'PASS')} passed, {r.failed} failed")
    return 1 if r.failed else 0


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="copy the subset into a new directory")
    c.add_argument("--src", required=True, help="the full EXL3 checkpoint directory (read only)")
    c.add_argument("--dst", required=True, help="new directory for the mini (must not exist)")
    c.add_argument("--include-visual", action="store_true", help="also copy the vision tower (model.visual.*)")
    c.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    v = sub.add_parser("verify", help="check a mini against its source")
    v.add_argument("--src", required=True)
    v.add_argument("--dst", required=True)
    v.add_argument("--include-visual", action="store_true", default=None,
                   help="expect the vision tower (default: what the mini's provenance file says)")
    v.add_argument("--jobs", type=int, default=4, help="hashing threads (default 4)")
    v.add_argument("--json-out", help="also write the check table as JSON")
    k = sub.add_parser("check-config", help="parse the mini with transformers and vLLM (in the vLLM image)")
    k.add_argument("--dst", required=True)
    k.add_argument("--src", help="also parse the source and compare")
    args = ap.parse_args(argv)
    try:
        return {"create": cmd_create, "verify": cmd_verify, "check-config": cmd_check_config}[args.cmd](args)
    except Fail as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
