#!/usr/bin/env python3
"""Build the Nav2 occupancy map directly from gazebo_worlds/tabletop.sdf.

Every <box> collision of a static model (walls, landmarks, table top + post) is
rasterized; dynamic models (cup) and the ground plane are skipped. The map frame
is therefore exactly the Gazebo world frame. A SLAM map was ~3 cm off and missed
the table top, which sits above the lidar.

Cells inside the arena walls are free, cells outside are unknown.

  python3 make_map_from_world.py            # -> maps/tabletop_nav.{pgm,yaml}
  python3 make_map_from_world.py --margin 0.05
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WORLD = ROOT / 'gazebo_worlds' / 'tabletop.sdf'
MAPS = ROOT / 'maps'
FREE, OCC, UNKNOWN = 254, 0, 205


def pose_of(elem):
    p = elem.find('pose') if elem is not None else None
    v = [float(x) for x in p.text.split()] if p is not None and p.text else []
    return (v + [0.0] * 6)[:6]


def compose(a, b):
    """2D compose of (x, y, z, r, p, yaw) poses; roll/pitch ignored (all boxes upright)."""
    c, s = math.cos(a[5]), math.sin(a[5])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2],
            0.0, 0.0, a[5] + b[5])


def boxes(world_file):
    root = ET.parse(world_file).getroot()
    out = []
    for model in root.iter('model'):
        name = model.get('name')
        static = (model.findtext('static') or '').strip().lower() in ('true', '1')
        if not static or name == 'ground':
            continue
        mpose = pose_of(model)
        for link in model.findall('link'):
            lpose = compose(mpose, pose_of(link))
            for col in link.findall('collision'):
                size = col.find('geometry/box/size')
                if size is None:
                    continue
                sx, sy, sz = (float(v) for v in size.text.split())
                cpose = compose(lpose, pose_of(col))
                out.append((name, cpose, sx, sy, sz))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', type=Path, default=WORLD)
    ap.add_argument('--out', default='tabletop_nav')
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--margin', type=float, default=0.0,
                    help='Grow every obstacle by this much (m)')
    a = ap.parse_args()

    bxs = boxes(a.world)
    walls = [b for b in bxs if b[0] == 'arena_walls']
    if not walls:
        raise SystemExit('No arena_walls model in the world')
    # Map covers the walls plus one cell.
    xmin = min(b[1][0] - b[2] / 2 for b in walls) - a.res
    xmax = max(b[1][0] + b[2] / 2 for b in walls) + a.res
    ymin = min(b[1][1] - b[3] / 2 for b in walls) - a.res
    ymax = max(b[1][1] + b[3] / 2 for b in walls) + a.res
    w, h = int(math.ceil((xmax - xmin) / a.res)), int(math.ceil((ymax - ymin) / a.res))
    # Inner free rectangle = between the walls' inner faces.
    inner = (max(b[1][0] + b[2] / 2 for b in walls if b[2] < b[3] and b[1][0] < 0),
             min(b[1][0] - b[2] / 2 for b in walls if b[2] < b[3] and b[1][0] > 0),
             max(b[1][1] + b[3] / 2 for b in walls if b[3] < b[2] and b[1][1] < 0),
             min(b[1][1] - b[3] / 2 for b in walls if b[3] < b[2] and b[1][1] > 0))

    img = Image.new('L', (w, h), UNKNOWN)
    px = img.load()
    for j in range(h):
        y = ymin + (h - 1 - j + 0.5) * a.res          # image row 0 is the top (max y)
        for i in range(w):
            x = xmin + (i + 0.5) * a.res
            if inner[0] <= x <= inner[1] and inner[2] <= y <= inner[3]:
                px[i, j] = FREE
            for _, (bx, by, _, _, _, yaw), sx, sy, _ in bxs:
                c, s = math.cos(-yaw), math.sin(-yaw)
                lx, ly = c * (x - bx) - s * (y - by), s * (x - bx) + c * (y - by)
                if abs(lx) <= sx / 2 + a.margin and abs(ly) <= sy / 2 + a.margin:
                    px[i, j] = OCC
                    break

    MAPS.mkdir(exist_ok=True)
    pgm, yml = MAPS / f'{a.out}.pgm', MAPS / f'{a.out}.yaml'
    img.save(pgm)
    yml.write_text(yaml.safe_dump({
        'image': pgm.name, 'mode': 'trinary', 'resolution': a.res,
        'origin': [round(xmin, 4), round(ymin, 4), 0.0], 'negate': 0,
        'occupied_thresh': 0.65, 'free_thresh': 0.196}, sort_keys=False))
    print(f'{len(bxs)} static boxes -> {pgm} ({w}x{h} @ {a.res} m, origin '
          f'({xmin:.2f}, {ymin:.2f}))')
    for name, p, sx, sy, sz in bxs:
        print(f'  {name:12} at ({p[0]:+.2f}, {p[1]:+.2f}) size {sx:.2f} x {sy:.2f} x {sz:.2f}')


if __name__ == '__main__':
    main()
