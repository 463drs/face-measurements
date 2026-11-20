import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PARQUET_SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("image_path", pa.string()),
    ("points", pa.list_(pa.list_(pa.float32(), 2))),
    ("schema", pa.string()),
    ("attributes", pa.string()),
])


@dataclass
class Sample:
    image_id: str
    image: np.ndarray
    points: np.ndarray
    schema: str
    attributes: dict


def write_parquet(rows: Iterable[dict], path: str | Path, batch_size: int = 256) -> int:
    """Rows are dicts with Sample fields; `image` holds encoded image bytes (jpg/png)."""
    path = Path(path)
    base = path.parent.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pq.ParquetWriter(path, PARQUET_SCHEMA) as writer:
        batch = []
        for row in rows:
            batch.append({
                "image_id": row["image_id"],
                "image_path": Path(os.path.relpath(Path(row["image_path"]).resolve(), base)).as_posix(),
                "points": np.asarray(row["points"], dtype=np.float32).reshape(-1, 2).tolist(),
                "schema": row["schema"],
                "attributes": json.dumps(row.get("attributes", {})),
            })
            if len(batch) == batch_size:
                writer.write_table(pa.Table.from_pylist(batch, PARQUET_SCHEMA))
                count += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, PARQUET_SCHEMA))
            count += len(batch)
    return count


class FaceDataset:
    def __init__(self, paths: str | Path | list[str | Path]):
        if not isinstance(paths, list):
            paths = [paths]
        tables = []
        for p in paths:
            table = pq.read_table(p, schema=PARQUET_SCHEMA)
            base = Path(p).resolve().parent
            resolved = [str(base / rel) for rel in table["image_path"].to_pylist()]
            idx = table.schema.get_field_index("image_path")
            tables.append(table.set_column(idx, "image_path", pa.array(resolved)))
        self.table = pa.concat_tables(tables)
    
    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> Sample:
        row = self.table.slice(idx, 1).to_pylist()[0]
        return Sample(
            image_id=row["image_id"],
            image=np.asarray(Image.open(row["image_path"]).convert("RGB")),
            points=np.asarray(row["points"], dtype=np.float32),
            schema=row["schema"],
            attributes=json.loads(row["attributes"]),
        )

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]
