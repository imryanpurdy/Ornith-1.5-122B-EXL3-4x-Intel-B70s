"""Read single tensors from a sharded safetensors checkpoint by exact byte range (read-only, no mmap of whole
shards): model.safetensors.index.json -> shard, the shard's JSON header -> [begin, end), os.pread of those bytes."""
from __future__ import annotations

import json
import os
import struct

import torch

_DT = {"I16": torch.int16, "I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
       "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8}


class Checkpoint:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._hdr: dict[str, tuple[dict, int]] = {}
        self.bytes_read = 0

    def config(self) -> dict:
        with open(os.path.join(self.dir, "config.json")) as f:
            c = json.load(f)
        return c.get("text_config", c)

    def names(self) -> list[str]:
        return list(self.weight_map)

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._hdr:
            fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
            try:
                (n,) = struct.unpack("<Q", os.pread(fd, 8, 0))
                self._hdr[shard] = (json.loads(os.pread(fd, n, 8)), 8 + n)
            finally:
                os.close(fd)
        return self._hdr[shard]

    def meta(self, name: str) -> dict:
        h, _ = self._header(self.weight_map[name])
        return h[name]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        h, base = self._header(shard)
        e = h[name]
        b0, b1 = e["data_offsets"]
        fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
        try:
            buf = bytearray(b1 - b0)
            mv, off = memoryview(buf), 0
            while off < len(buf):                          # pread may return short reads for large ranges
                chunk = os.pread(fd, min(len(buf) - off, 1 << 30), base + b0 + off)
                if not chunk:
                    raise IOError(f"short read for {name}")
                mv[off:off + len(chunk)] = chunk
                off += len(chunk)
        finally:
            os.close(fd)
        self.bytes_read += len(buf)
        t = torch.frombuffer(buf, dtype=_DT[e["dtype"]]) if len(buf) else torch.empty(0, dtype=_DT[e["dtype"]])
        return t.reshape(e["shape"])

    def exl3(self, module: str) -> dict[str, torch.Tensor]:
        return {p: self.get(f"{module}.{p}") for p in ("trellis", "suh", "svh", "mul1")}
