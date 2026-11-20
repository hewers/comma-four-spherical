"""Build per-camera masks of outside scenery from temporal variation across drives."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess as sp

import cv2
import imageio_ffmpeg
import numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path('rig/masks')
# Full-resolution polygons. For the cabin this removes seats, people and door
# panels; the temporal test then removes pillars, roof, mirror and visor.
KEEP = {
    'e': [[(40, 400), (60, 230), (150, 90), (330, 5), (1010, 5), (1190, 70), (1285, 220), (1312, 395),
           (1200, 452), (960, 478), (680, 484), (400, 478), (160, 448)]],
    'f': [[(0, 0), (1344, 0), (1344, 610), (0, 610)]],
    'd': [[(0, 0), (1344, 0), (1344, 560), (1300, 525), (1200, 432), (985, 300), (962, 180), (905, 180),
           (905, 228), (470, 228), (470, 180), (405, 185), (420, 300), (240, 362), (95, 492), (60, 520), (0, 560)]],
}


def grab(path):
    """Blurred absolute difference of two frames 100 ms apart, and the first frame."""
    out = sp.run([FF, '-v', 'error', '-threads', '1', '-i', str(path), '-vf', "select='eq(n\\,20)+eq(n\\,22)'",
                  '-fps_mode', 'passthrough', '-frames:v', '2', '-pix_fmt', 'gray', '-f', 'rawvideo', '-'],
                 capture_output=True).stdout
    if len(out) != 2*760*1344:
        return None
    a, b = np.frombuffer(out, np.uint8).reshape(2, 760, 1344).astype(np.float32)
    if a.mean() < 30:
        return None
    diff = cv2.GaussianBlur(np.abs(a-b), (0, 0), 3)/a.mean()
    return diff, a.astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segs = sorted(p.parent for p in Path('.').glob('*--*/dcamera.hevc'))[::2]
    for k in 'edf':
        with ProcessPoolExecutor(12) as pool:
            found = [f for f in pool.map(grab, [s/f'{k}camera.hevc' for s in segs]) if f is not None]
        # Keep pairs with real motion somewhere in the frame, then ask how often each pixel changes.
        moving = [d for d, _ in found if np.quantile(d, .9) > .02]
        frames = [f for _, f in found]
        stack = np.array(moving)
        cv = (stack > (.035 if k == 'd' else .012)).mean(axis=0)
        keep = np.zeros((760, 1344), np.uint8)
        for poly in KEEP[k]:
            cv2.fillPoly(keep, [np.array(poly, np.int32)], 255)
        thresh = {'e': .22, 'f': .22, 'd': .25}[k]
        mask = ((cv > thresh).astype(np.uint8)*255) & keep
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        mask = cv2.erode(mask, np.ones((13, 13), np.uint8))
        cv2.imwrite(str(OUT/f'{k}.png'), mask)
        view = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_GRAY2BGR)
        view[mask == 0] //= 3
        heat = cv2.applyColorMap(np.clip(cv*255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f'previews/rig/mask-{k}.jpg', np.vstack([view, heat])[::2, ::2])
        print(k, len(stack), 'frames; mask fraction', round(float(mask.mean()/255), 3), flush=True)


if __name__ == '__main__':
    main()
