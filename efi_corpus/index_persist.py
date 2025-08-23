from pathlib import Path
from efi_core.protocols import AnnIndex


def save(index: AnnIndex, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    index.persist(path)


def load(index: AnnIndex, path: Path) -> AnnIndex:
    index.load(Path(path))
    return index


