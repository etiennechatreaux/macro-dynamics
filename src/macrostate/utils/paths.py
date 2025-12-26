from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_data_path(subpath: str = "") -> Path:
    """Get path relative to data directory."""
    root = get_project_root()
    data_dir = root / "data"
    if subpath:
        return data_dir / subpath
    return data_dir

