from pathlib import Path
from glob import glob
from typing import Iterable, List, Optional
import numpy as np


def get_relpaths_from_dir(dir_path: Path, *, extension: str) -> List[Path]:
    """
    Return *relative* paths of every file with the given extension that
    lives under ``dir_path`` (recursively).
    """
    return [
        p.relative_to(dir_path) 
        for p in dir_path.rglob(f'*.{extension}') 
        if p.is_file()
    ]


def get_relpaths_matching_subdirs(
    dir_path: Path,
    *,
    extension: str,
    include_subdirs: Optional[Iterable[str]] = None,
    include_root: bool = True,
) -> List[Path]:
    """
    Like ``get_relpaths_from_dir`` but keep only the files whose *immediate
    parent directory* is listed in ``include_subdirs``.

    ``include_subdirs`` may contain paths such as::

        ['Category1', 'Category1/TopicA', 'Category2']

    All paths are interpreted **relative to ``dir_path``**.

    Setting ``include_subdirs=None`` means “include *all* sub-dirs”.
    ``include_root`` controls whether files that sit directly in ``dir_path``
    itself are kept.
    """
    all_rel = get_relpaths_from_dir(dir_path, extension=extension)

    if not include_subdirs:
        return (
            all_rel
            if include_root
            else [p for p in all_rel if p.parent != Path(".")]
        )

    allowed_dirs = [Path(s).with_suffix("").relative_to(".") for s in include_subdirs]

    def _in_allowed_tree(path: Path) -> bool:
        """True if the file lives in / under one of the allowed dirs."""
        parent = path.parent
        return any(parent == d or parent.is_relative_to(d) for d in allowed_dirs)

    return [
        p
        for p in all_rel
        if (include_root and p.parent == Path(".")) or _in_allowed_tree(p)
    ]

def _get_valid_filepaths_by_ext_set(dirpath: Path, *,
                                    exts: set[str]):
    all_files = [p.relative_to(dirpath)
                 for p in Path(dirpath).glob("**/*")
                 if p.suffix in exts]
    return all_files


def _get_shortest_path_by_filename(relpaths_list: list[Path]) -> dict[str, Path]:
    # get filename w/ ext only:
    all_file_names_list = [f.name for f in relpaths_list]

    # get indices of dupe 'filename w/ ext':
    _, inverse_ix, counts = np.unique(
        np.array(all_file_names_list),
        return_inverse=True,
        return_counts=True,
        axis=0)
    dupe_names_ix = np.where(counts[inverse_ix] > 1)[0]

    # get shortest paths via mask:
    shortest_paths_arr = np.array(all_file_names_list, dtype=object)
    shortest_paths_arr[dupe_names_ix] = np.array(
        [str(fpath)
         for fpath in relpaths_list])[dupe_names_ix]
    return {fn: path for fn, path in zip(shortest_paths_arr, relpaths_list)}
