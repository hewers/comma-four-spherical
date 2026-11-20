"""Render equirectangular JPG stills for a calibration, optionally with a level/heading grid."""
import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

from rig_collect import decode
from stitch import maps, timing

DEFAULT = ['000000e9--6975f83496--34:214', '000000da--fafd43926e--50:300', '000000ea--f23abe10b0--10:300',
           '000000dd--d304160507--50:0', '000000e9--6975f83496--8:600', '000000dd--d304160507--44:600']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('calibration', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--width', type=int, default=2048)
    ap.add_argument('--grid', action='store_true', help='draw the horizon and the forward/side/rear meridians')
    ap.add_argument('--order', default='def', help='draw order; wide blends 50/50 with cabin where both see, narrow overwrites')
    ap.add_argument('scenes', nargs='*', default=DEFAULT, help='SEGMENT:WIDE_FRAME')
    a = ap.parse_args()
    cfg = json.loads(a.calibration.read_text())
    w, h = a.width, a.width//2
    lon, lat = np.meshgrid(((np.arange(w, dtype='float32')+.5)/w-.5)*2*np.pi, ((np.arange(h, dtype='float32')+.5)/h-.5)*np.pi)
    sphere = np.stack((np.sin(lon)*np.cos(lat), np.sin(lat), np.cos(lon)*np.cos(lat)), axis=-1)
    warps = {k: maps(cfg[k], sphere) for k in 'edf'}
    a.output.mkdir(parents=True, exist_ok=True)
    for scene in a.scenes:
        name, frame = scene.split(':')
        seg = Path(name)
        ts, offset = timing(seg)
        t = ts['e'][int(frame)]
        out = np.zeros((h, w, 3), np.uint8)
        both = warps['e'][1] & warps['d'][1]
        for k in a.order:
            i = int(np.argmin(np.abs(ts[k]-t)))
            gray = decode(seg/f'{k}camera.hevc', [i])[i]
            xy, mask = warps[k]
            warped = cv2.remap(gray, *xy, cv2.INTER_LINEAR)
            tint = {'e': (1., 1., 1.), 'd': (1., .93, .93), 'f': (.93, 1., .93)}[k] if a.grid else (1., 1., 1.)
            layer = (warped[..., None]*np.array(tint)).astype(np.uint8)
            if k == 'e' and a.order.index('d') < a.order.index('e'):
                # 50/50 where wide and cabin overlap
                layer[both] = (layer[both].astype(np.uint16) + out[both]) >> 1
            np.copyto(out, layer, where=mask[..., None])
        if a.grid:
            out[h//2-1:h//2+1, ::4] = (255, 255, 0)
            for deg in (-135, -90, -45, 0, 45, 90, 135):
                x = int((deg/360+.5)*w)
                out[::6, x] = (0, 255, 255)
        stamp = dt.datetime.fromtimestamp((int(t)+offset)/1e9, dt.UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        cv2.putText(out, stamp, (w//2-300, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        path = a.output/f'{name}-{frame}.jpg'
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(path, stamp, flush=True)


if __name__ == '__main__':
    main()
