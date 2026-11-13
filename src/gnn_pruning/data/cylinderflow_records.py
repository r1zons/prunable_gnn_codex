"""Indexed, frame-level reads from the checksum-verified official CFD TFRecords."""

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np
import torch

from .cylinderflow import Mesh, mesh_from_triangles


# GCS generations and object MD5s are verified during staging.
OBJECTS = {
    "train": (13645805387, 1000, 1622067156357701),
    "valid": (1363987289, 100, 1622065354488825),
}
FRAMES = 600


def _varint(stream):
    value = 0
    for shift in range(0, 70, 7):
        byte = stream.read(1)
        if not byte:
            break
        number = byte[0]
        value |= (number & 127) << shift
        if number < 128:
            return value
    raise ValueError("Invalid protobuf varint")


def _fields(stream, start, end):
    """Yield length-delimited fields in the official Example encoding."""
    while start < end:
        stream.seek(start)
        key, size = _varint(stream), _varint(stream)
        start = stream.tell()
        stop = start + size
        if key < 8 or key & 7 != 2 or stop > end:
            raise ValueError("Invalid TensorFlow Example field")
        yield key >> 3, start, stop
        start = stop


def _only_field(stream, start, end):
    (tag, offset, stop), = _fields(stream, start, end)
    if tag != 1:
        raise ValueError("Expected bytes-valued Example field")
    return offset, stop


def _example_index(stream, start, end):
    fields_start, fields_end = _only_field(stream, start, end)
    result = {}
    for tag, offset, stop in _fields(stream, fields_start, fields_end):
        if tag != 1:
            raise ValueError("Expected Features map entry")
        entries = {k: (a, b) for k, a, b in _fields(stream, offset, stop)}
        if set(entries) != {1, 2}:
            raise ValueError("Invalid Features map entry")
        a, b = entries[1]
        if b - a > 32:
            raise ValueError("Invalid feature key")
        stream.seek(a)
        name = stream.read(b - a).decode("ascii")
        if name in result:
            raise ValueError("Duplicate trajectory feature")
        a, b = _only_field(stream, *entries[2])
        result[name] = _only_field(stream, a, b)
    if set(result) != {"cells", "mesh_pos", "node_type", "velocity", "pressure"}:
        raise ValueError("Unexpected CylinderFlow fields")
    return result


def _array(stream, offset, size, dtype):
    stream.seek(offset)
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Incomplete CylinderFlow array")
    return np.frombuffer(value, dtype=dtype).copy()


class CylinderFlowRecords:
    """Index official train or valid data and keep a bounded cache of static meshes."""

    def __init__(self, path: Path, split: str):
        size, count, generation = OBJECTS[split]
        self.split = split
        self.generation = generation
        self._stream = path.open("rb")
        self.records = []
        if path.stat().st_size != size:
            raise ValueError(f"Unexpected {split} object length")
        offset = 0
        while offset < size:
            self._stream.seek(offset)
            header = self._stream.read(12)
            length, _ = struct.unpack("<QI", header)
            start, end = offset + 12, offset + 12 + length
            if length == 0 or end + 4 > size:
                raise ValueError(f"Truncated {split} TFRecord")
            fields = _example_index(self._stream, start, end)
            widths = {name: b - a for name, (a, b) in fields.items()}
            nodes, remainder = divmod(widths["mesh_pos"], 8)
            if (not nodes or remainder or not widths["cells"] or widths["cells"] % 12
                    or widths["node_type"] != nodes * 4
                    or widths["velocity"] != FRAMES * nodes * 8
                    or widths["pressure"] != FRAMES * nodes * 4):
                raise ValueError(f"Invalid field dimensions in {split}:{len(self.records)}")
            self.records.append((nodes, fields))
            offset = end + 4
        if len(self.records) != count:
            raise ValueError(f"Unexpected {split} trajectory count")
        self.mesh = lru_cache(maxsize=128)(self.mesh)

    def __len__(self):
        return len(self.records)

    def geometry(self, ordinal: int):
        """Read validated original triangle cells, coordinates and node types."""
        nodes, fields = self.records[ordinal]
        arrays = {}
        for name, dtype in (("cells", "<i4"), ("mesh_pos", "<f4"), ("node_type", "<i4")):
            offset, end = fields[name]
            arrays[name] = _array(self._stream, offset, end - offset, dtype)
        cells = arrays["cells"].reshape(-1, 3)
        pos = arrays["mesh_pos"].reshape(nodes, 2)
        types = arrays["node_type"]
        if (cells.min() < 0 or cells.max() >= nodes or np.any(cells == cells[:, [1, 2, 0]])
                or not np.isfinite(pos).all() or types.min() < 0 or types.max() >= 9
                or not np.isin(types, [0, 5]).any()):
            raise ValueError(f"Invalid mesh {self.split}:{ordinal}")
        return torch.from_numpy(cells).long(), torch.from_numpy(pos), torch.from_numpy(types).long()

    def mesh(self, ordinal: int) -> Mesh:
        """Read and deduplicate the static graph of one trajectory."""
        return mesh_from_triangles(*self.geometry(ordinal))

    def velocity(self, ordinal: int, frame: int, count: int) -> torch.Tensor:
        """Read only the contiguous frame window needed by the caller."""
        if not 0 <= frame < frame + count <= FRAMES:
            raise ValueError("Frame window out of bounds")
        nodes, fields = self.records[ordinal]
        offset = fields["velocity"][0] + frame * nodes * 8
        values = _array(self._stream, offset, count * nodes * 8, "<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite velocity in {self.split}:{ordinal}")
        return torch.from_numpy(values.reshape(count, nodes, 2))
