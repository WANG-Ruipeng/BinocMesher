"""Original effective SourceVID is a prerequisite, not a geometric claim."""
import numpy as np


def original_identity_receipt(snapshot):
    if snapshot.get('identity_status') != 1:
        raise ValueError('Original SourceVID observation is not ready.')
    if (snapshot.get('source_vid_encoding_version') != 2
            or snapshot.get('source_vid_encoding') != 'ORIGINAL_EFFECTIVE_SOURCE_VID'):
        raise ValueError('Original SourceVID encoding is unverified; normalized merger keys cannot establish admission.')
    shifts = snapshot.get('source_vid_shifts')
    if (not isinstance(shifts, np.ndarray) or shifts.shape != (5, 4)
            or shifts.dtype != np.int32 or shifts.flags.writeable):
        raise ValueError('Original SourceVID requires an immutable int32 five-element shift receipt.')
    return {'version': 2, 'encoding': 'ORIGINAL_EFFECTIVE_SOURCE_VID',
            'shifts_order': ['node0', 'group0', 'node1', 'group1'],
            'per_element_shifts': shifts.tolist()}
