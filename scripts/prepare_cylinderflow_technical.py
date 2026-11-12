"""Stage record zero of the official train and valid splits without downloading whole splits."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


BASE = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow"
GENERATIONS = {"meta.json": 1622065139906737, "train": 1622067156357701,
               "valid": 1622065354488825}


def example_type():
    """Describe the bytes-only subset of the TensorFlow Example wire schema."""
    schema = descriptor_pb2.FileDescriptorProto(name="cylinderflow_example.proto")
    for name, fields in (
        ("BytesList", [("value", 3, 12, "")]),
        ("Feature", [("bytes_list", 1, 11, ".BytesList")]),
        ("Entry", [("key", 1, 9, ""), ("value", 1, 11, ".Feature")]),
        ("Features", [("feature", 3, 11, ".Entry")]),
        ("Example", [("features", 1, 11, ".Features")]),
    ):
        message = schema.message_type.add(name=name)
        for number, (field, label, kind, type_name) in enumerate(fields, 1):
            entry = message.field.add(name=field, number=number, label=label, type=kind)
            if type_name:
                entry.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Example"))


def crc_table():
    """Build the reflected Castagnoli lookup table used by TFRecord."""
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return tuple(table)


CRC_TABLE = crc_table()


def masked_crc(data):
    """Return the masked CRC32C stored by TFRecord."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ CRC_TABLE[(crc ^ byte) & 255]
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def fetch(name, generation, byte_range=None):
    """Read one pinned GCS object or an explicitly verified bounded range."""
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    with urlopen(Request(f"{BASE}/{name}?generation={generation}", headers=headers),
                 timeout=45) as response:
        if response.headers.get("x-goog-generation") != str(generation):
            raise ValueError(f"Unexpected generation for {name}")
        if byte_range:
            start, stop = byte_range
            if response.status != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {start}-{stop}/"):
                raise ValueError(f"Server did not honor bounded range for {name}")
            data = response.read(stop - start + 2)
            if len(data) != stop - start + 1:
                raise ValueError(f"Incomplete range for {name}")
            return data
        data = response.read(8193)
        if len(data) > 8192:
            raise ValueError("Unexpectedly large dataset metadata")
        return data


def stage(root):
    """Write official raw records, converted tensors and their provenance."""
    root.mkdir(parents=True, exist_ok=False)
    meta_bytes = fetch("meta.json", GENERATIONS["meta.json"])
    meta = json.loads(meta_bytes)
    (root / "meta.json").write_bytes(meta_bytes)
    manifest = {}
    for split in ("train", "valid"):
        name = f"{split}.tfrecord"
        generation = GENERATIONS[split]
        header = fetch(name, generation, (0, 11))
        length, header_crc = struct.unpack("<QI", header)
        if header_crc != masked_crc(header[:8]) or not 0 < length <= 64 * 1024**2:
            raise ValueError(f"Invalid record header for {split}")
        record = fetch(name, generation, (12, 15 + length))
        if struct.unpack("<I", record[-4:])[0] != masked_crc(memoryview(record)[:-4]):
            raise ValueError(f"Invalid payload CRC for {split}")
        raw = root / f"{split}-000000.tfrecord"
        raw.write_bytes(header + record)
        example = example_type().FromString(record[:-4])
        features = {entry.key: entry.value.bytes_list.value for entry in example.features.feature}
        arrays = {}
        for key in ("cells", "mesh_pos", "node_type", "velocity"):
            field = meta["features"][key]
            dtype = np.dtype(field["dtype"]).newbyteorder("<")
            values = np.frombuffer(b"".join(features[key]), dtype=dtype).reshape(field["shape"])
            arrays[key] = values[0] if field["type"] == "static" else values
        converted = root / f"{split}-000000.npz"
        np.savez(converted, split=split, ordinal=0, **arrays)
        manifest[split] = {
            "record_id": f"{split}:0", "source": f"{BASE}/{name}?generation={generation}",
            "record_bytes": raw.stat().st_size, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "npz_sha256": hashlib.sha256(converted.read_bytes()).hexdigest(),
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
    (root / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    stage(parser.parse_args().output)
