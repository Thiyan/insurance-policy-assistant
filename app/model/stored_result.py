from dataclasses import dataclass


@dataclass
class StoreResult:
    collection_name: str
    db_path: str
    chunks_stored: int
    was_replaced: bool
