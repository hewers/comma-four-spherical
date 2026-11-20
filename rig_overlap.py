"""Collect wide/narrow SIFT correspondences on simultaneous frames for the rig fit."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from cameras import match


def work(seg):
    cv2.setNumThreads(1)
    try:
        found = match(Path(seg), cv2.SIFT_create(nfeatures=1500), cv2.BFMatcher(cv2.NORM_L2))
    except (OSError, ValueError, StopIteration, cv2.error):
        return seg, None
    return seg, found


def main():
    segs = sorted(f.stem for f in Path('rig/tracks').glob('*.npz'))
    names, wide, narrow = [], [], []
    with ProcessPoolExecutor(12) as pool:
        for seg, found in pool.map(work, segs):
            if found is None:
                continue
            a, b = found
            keep = np.linspace(0, len(a)-1, min(80, len(a))).astype(int)
            names += [seg]*len(keep); wide.append(a[keep]); narrow.append(b[keep])
    np.savez_compressed('rig/overlap.npz', segment=np.array(names), wide=np.concatenate(wide), narrow=np.concatenate(narrow))
    print(len(set(names)), 'segments,', len(names), 'matches')


if __name__ == '__main__':
    main()
