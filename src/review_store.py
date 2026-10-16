"""Validated review sidecars; independent of inference and the scoring outputs."""
import copy
import json
import math
import os
import threading
import uuid
from datetime import datetime

import numpy as np

STAGES = ('inspect', 'submitted', 'prediction', 'expert')
MODALITIES = ('t1', 't1c', 't2', 'flair')
REVIEW_LOCK = threading.Lock()


class ReviewConflict(ValueError):
    pass


def empty_review():
    return dict(schema_version=1, revision=0, bookmarks=[], attempts=[], active_attempt_id=None)


def load_review(folder):
    path = folder / 'review.json'
    return json.loads(path.read_text()) if path.exists() else empty_review()


def text(value, limit, name, required=False):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f'{name} must be {"nonempty " if required else ""}text of at most {limit} characters.')
    return value


def timestamp(value):
    text(value, 40, 'Timestamp', True)
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('Invalid timestamp.')
    return value


def identity(value):
    text(value, 64, 'ID', True)
    return value


def vector(value):
    if not isinstance(value, list) or len(value) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('Coordinates must contain three finite numbers.')
    return value


def validate_bookmark(item, metadata):
    if not isinstance(item, dict):
        raise ValueError('Invalid bookmark.')
    world = vector(item.get('world_mm'))
    vector(item.get('voxel'))
    affine, shape = metadata.get('affine'), metadata.get('shape')
    if affine is None or shape is None:
        raise ValueError('Original image geometry is unavailable; cannot save a spatial bookmark.')
    voxel = (np.linalg.inv(np.asarray(affine)) @ np.array([*world, 1.0]))[:3]
    if not np.isfinite(voxel).all() or any(x < -1e-5 or x > size - 1 + 1e-5 for x, size in zip(voxel, shape)):
        raise ValueError('Bookmark lies outside the original image volume.')
    modality = item.get('modality')
    if modality not in metadata['available_modalities']:
        raise ValueError('Bookmark modality is unavailable for this analysis.')
    view, overlay, opacity = item.get('view'), item.get('overlay'), item.get('opacity')
    if type(view) is not int or view not in range(5) or overlay not in ('none', 'segmentation', 'expert'):
        raise ValueError('Invalid bookmark viewer settings.')
    if overlay == 'expert' and not metadata['has_expert']:
        raise ValueError('No expert reference available.')
    if type(opacity) not in (int, float) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Opacity must be between zero and one.')
    return dict(id=identity(item.get('id')), title=text(item.get('title'), 100, 'Title', True),
                note=text(item.get('note', ''), 2000, 'Note'), world_mm=world,
                voxel=voxel.tolist(), modality=modality, view=view, overlay=overlay, opacity=opacity,
                created_at=timestamp(item.get('created_at')), updated_at=timestamp(item.get('updated_at')))


def bookmarks(items, metadata):
    if not isinstance(items, list) or len(items) > 500:
        raise ValueError('At most 500 bookmarks are allowed.')
    result = [validate_bookmark(b, metadata) for b in items]
    if len({b['id'] for b in result}) != len(result):
        raise ValueError('Duplicate bookmark IDs.')
    return result


def validate_review(data, previous, metadata):
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Unsupported review schema.')
    if type(data.get('revision')) is not int or data['revision'] != previous['revision']:
        raise ReviewConflict('Review changed in another tab. Reload the saved review before retrying.')
    attempts = data.get('attempts')
    if not isinstance(attempts, list) or len(attempts) > 100:
        raise ValueError('At most 100 attempts are allowed.')
    old = {a['id']: a for a in previous['attempts']}
    validated = []
    for a in attempts:
        if not isinstance(a, dict):
            raise ValueError('Invalid attempt.')
        aid = identity(a.get('id'))
        stage = a.get('stage')
        if stage not in STAGES:
            raise ValueError('Invalid attempt stage.')
        prior = old.get(aid)
        if prior is None and stage != 'inspect':
            raise ValueError('A new attempt must begin at Inspect.')
        if prior and STAGES.index(stage) - STAGES.index(prior['stage']) not in (0, 1):
            raise ValueError('Reveal stages must advance one step at a time.')
        if stage == 'expert' and not metadata['has_expert']:
            raise ValueError('No expert reference available.')
        exposure = a.get('exposure')
        if exposure not in ('unseen', 'practice_replay', 'unknown'):
            raise ValueError('Invalid exposure status.')
        current = dict(id=aid, stage=stage, exposure=exposure, created_at=timestamp(a.get('created_at')),
                       updated_at=timestamp(a.get('updated_at')), bookmarks=bookmarks(a.get('bookmarks'), metadata),
                       submitted_at=a.get('submitted_at'), submitted_bookmarks=a.get('submitted_bookmarks'),
                       no_areas=a.get('no_areas', False), reflection=text(a.get('reflection', ''), 2000, 'Reflection'))
        if prior:
            for key in ('created_at', 'exposure'):
                if current[key] != prior[key]:
                    raise ValueError('Attempt identity cannot be changed.')
        if any(b['overlay'] != 'none' for b in current['bookmarks']):
            raise ValueError('Blind observations must use scan-only viewing.')
        if stage == 'inspect':
            if current['submitted_at'] is not None or current['submitted_bookmarks'] is not None or current['no_areas'] is not False:
                raise ValueError('An inspection cannot contain a submission.')
        else:
            timestamp(current['submitted_at'])
            if type(current['no_areas']) is not bool:
                raise ValueError('Invalid empty-submission acknowledgement.')
            # Compare the supplied snapshot with the validated bookmarks; clients cannot inject separate answers.
            if current['submitted_bookmarks'] != a['bookmarks']:
                raise ValueError('Submitted snapshot must match the observations.')
            current['submitted_bookmarks'] = copy.deepcopy(current['bookmarks'])
            if not current['bookmarks'] and current['no_areas'] is not True:
                raise ValueError('Explicitly acknowledge a submission with no areas bookmarked.')
            if current['bookmarks'] and current['no_areas']:
                raise ValueError('An empty submission cannot contain bookmarks.')
        if prior and prior['stage'] != 'inspect':
            for key in ('bookmarks', 'submitted_bookmarks', 'submitted_at', 'no_areas'):
                if current[key] != prior[key]:
                    raise ValueError('Submitted observations are immutable.')
        validated.append(current)
    if len({a['id'] for a in validated}) != len(validated) or not set(old).issubset({a['id'] for a in validated}):
        raise ValueError('Attempts cannot be duplicated or deleted.')
    active = data.get('active_attempt_id')
    if active is not None and active not in {a['id'] for a in validated}:
        raise ValueError('Active attempt does not exist.')
    return dict(schema_version=1, revision=previous['revision'] + 1,
                bookmarks=bookmarks(data.get('bookmarks'), metadata), attempts=validated, active_attempt_id=active)


def save_review(folder, data, metadata):
    with REVIEW_LOCK:
        result = validate_review(data, load_review(folder), metadata)
        temporary = folder / f'.review-{uuid.uuid4().hex}.tmp'
        try:
            with temporary.open('w') as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, folder / 'review.json')
        finally:
            temporary.unlink(missing_ok=True)
        return result
