"""Shared helpers for reading camera frames and unprojecting pixels to rays."""
import cv2
import numpy as np

from stitch import imageio_ffmpeg, unit


def first_image(seg, key):
    """First frame of one camera as grayscale."""
    reader = imageio_ffmpeg.read_frames(str(seg/f'{key}camera.hevc'), input_params=['-threads', '1'])
    try:
        w, h = next(reader)['size']
        rgb = np.frombuffer(next(reader), np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    finally:
        reader.close()


def match(seg, sift, matcher):
    """SIFT correspondences between simultaneous wide and narrow frames (wide xy, narrow xy)."""
    # Enlarge the wide camera's middle to about the narrow camera's scale.
    wide, narrow = (first_image(seg, key) for key in 'ef')
    h, w = wide.shape
    if wide.shape != narrow.shape or (w, h) != (1344, 760):
        raise ValueError('unexpected image geometry')
    x, y = w//2-300, h//2-180
    crop = cv2.resize(wide[y:y+360, x:x+600], None, fx=2, fy=2)
    ka, da = sift.detectAndCompute(crop, None)
    kb, db = sift.detectAndCompute(narrow, None)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = [m for pair in matcher.knnMatch(da, db, k=2) if len(pair) == 2
             for m, n in [pair] if m.distance < .72*n.distance]
    if len(pairs) < 20:
        return None
    a = np.float32([ka[m.queryIdx].pt for m in pairs])/2 + [x, y]
    b = np.float32([kb[m.trainIdx].pt for m in pairs])
    # A homography rejects descriptor mismatches and selects a dominant
    # depth plane; downstream ray tests use a robust angular loss.
    _, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 4.)
    if mask is None or mask.sum() < 20:
        return None
    keep = mask.ravel().astype(bool)
    return a[keep], b[keep]


def rays(points, focal, equidistant, center=(672, 380), radial=(0., 0.)):
    """Unproject pixels to unit camera rays; the inverse of what stitch.maps() renders."""
    p = (points-center)/focal
    if equidistant:
        r = np.linalg.norm(p, axis=1)
        t = r * (1 + radial[0]*r*r + radial[1]*r**4)
        s = np.divide(np.sin(t), r, out=np.ones_like(r), where=r > 0)
        return np.column_stack((p*s[:, None], np.cos(t)))
    return unit(np.column_stack((p, np.ones(len(p)))))
