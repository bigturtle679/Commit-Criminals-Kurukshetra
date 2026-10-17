"""Accept one .zip of a patient's MRI scans and work out which file is which.

Recognises BraTS 2018-2021 names (_t1, _t1ce, _t2, _flair, _seg), BraTS 2023+
names (-t1n, -t1c, -t2w, -t2f, -seg) and common variants (T1Gd, T1post, mask,
label, ground truth). Members are never extracted by their own path, so a
crafted archive cannot write outside the job folder.
"""
import re
import zipfile
from pathlib import PurePosixPath

ROLES = ('t1c', 't1', 't2', 'flair', 'expert')
MAX_MEMBERS = 200
MAX_TOTAL_BYTES = 2 * 1024 ** 3
MAX_FILE_BYTES = 600 * 1024 ** 2

_EXPERT = {'seg', 'segmentation', 'mask', 'label', 'labels', 'gt', 'groundtruth', 'annotation', 'truth'}
_FLAIR = {'flair', 't2f', 't2flair'}
_T1C = {'t1c', 't1ce', 't1gd', 't1post', 't1contrast', 't1wce', 'ce', 'gd', 'contrast', 'postcontrast'}
_T2 = {'t2', 't2w'}
_T1 = {'t1', 't1n', 't1w', 't1native', 'native'}


def _stem(name):
    base = PurePosixPath(name).name.lower()
    for ext in ('.nii.gz', '.nii'):
        if base.endswith(ext):
            return base[:-len(ext)]
    return None


def classify(name):
    """Return the role for one NIfTI file name, or None if it is not recognisable."""
    stem = _stem(name)
    if stem is None:
        return None
    tokens = [t for t in re.split(r'[^a-z0-9]+', stem) if t]
    joined = {a + b for a, b in zip(tokens, tokens[1:])}  # t1_ce -> t1ce, ground_truth -> groundtruth
    words = set(tokens) | joined
    if words & _EXPERT:
        return 'expert'
    if words & _FLAIR or any('flair' in t for t in tokens):
        return 'flair'
    if words & _T1C:
        return 't1c'
    if words & _T2:
        return 't2'
    if words & _T1:
        return 't1'
    return None


def detect(archive: zipfile.ZipFile):
    """Map each role to one archive member. Raises ValueError with a readable message."""
    members = [m for m in archive.infolist() if not m.is_dir()]
    if any('..' in PurePosixPath(m.filename).parts or m.filename.startswith('/') for m in members):
        raise ValueError('Unsafe path traversal detected in zip archive.')
    if len(members) > MAX_MEMBERS:
        raise ValueError(f'Zip contains {len(members)} entries (limit is {MAX_MEMBERS}); expected one patient (5 files).')
    if sum(m.file_size for m in members) > MAX_TOTAL_BYTES:
        raise ValueError('Zip expands to more than 2 GB; upload one patient at a time.')
    found = {}
    for m in members:
        parts = PurePosixPath(m.filename).parts
        if any(p.startswith('.') or p == '__MACOSX' for p in parts):
            continue
        role = classify(m.filename)
        if role is None:
            continue
        if m.file_size > MAX_FILE_BYTES:
            raise ValueError(f'{PurePosixPath(m.filename).name} is larger than 600 MB.')
        found.setdefault(role, []).append(m)
    duplicates = {r: [PurePosixPath(m.filename).name for m in ms] for r, ms in found.items() if len(ms) > 1}
    if duplicates:
        detail = '; '.join(f"{r.upper()}: {', '.join(names)}" for r, names in duplicates.items())
        raise ValueError(f'More than one file matched the same scan type ({detail}). Zip one patient at a time.')
    missing = [r.upper() for r in ('t1c', 't1', 't2', 'flair') if r not in found]
    if missing:
        seen = ', '.join(sorted(PurePosixPath(m.filename).name for m in members)) or 'no files'
        raise ValueError(f"Could not find {', '.join(missing)} in the zip. Files seen: {seen}. "
                         'Name files with t1, t1ce/t1c, t2 and flair (and seg for the expert mask).')
    return {role: ms[0] for role, ms in found.items()}
