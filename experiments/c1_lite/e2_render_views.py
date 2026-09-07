"""Predeclared cameras and scientific image displays for fixed E2 only."""
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WIDTH, HEIGHT = 640, 360


def number(x):
    return float(Fraction(x['numerator'], x['denominator']))


def window_points(source):
    return np.asarray([[number(x) for x in vertex['position']]
        for point in source['breakpoint_points'] for vertex in point['boundary']], np.float64)


def camera_at(index):
    return {'c2w': np.asarray([[1, 0, 0, 0], [0, 0, 1, .3*float(index)],
                              [0, -1, 0, 3], [0, 0, 0, 1]], np.float64),
            'K': np.asarray([[1000, 0, WIDTH/2], [0, 1000, HEIGHT/2], [0, 0, 1]], np.float64),
            'width': WIDTH, 'height': HEIGHT, 'frame_index_continuous': float(index)}


def diagnostic_camera(source):
    points = window_points(source)
    lo, hi = points.min(0), points.max(0)
    target = (lo+hi)/2
    span = float(np.max(hi[:2]-lo[:2]))
    offset = np.asarray([1., -1.5, 2.5])
    eye = target+5*span*offset/np.linalg.norm(offset)
    forward = target-eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4); c2w[:3, :3] = np.column_stack((right, down, forward)); c2w[:3, 3] = eye
    return {'c2w': c2w, 'K': camera_at(0)['K'], 'width': WIDTH, 'height': HEIGHT,
            'target': target.tolist(), 'span': span,
            'reference_bounds_xy': [float(lo[0]-span/2), float(hi[0]+span/2),
                                    float(lo[1]-span/2), float(hi[1]+span/2)],
            'selection': 'frozen boundary-union bbox only; no treatment output used'}


def project(points, camera):
    points = np.asarray(points, np.float64)
    transform = camera['c2w']
    local = (points-transform[:3, 3]) @ transform[:3, :3]
    homogeneous = local @ camera['K'].T
    if np.any(local[:, 2] <= 1e-6):
        raise ValueError('Frozen ROI crosses the near plane; do not silently retune camera.')
    return homogeneous[:, :2]/homogeneous[:, 2, None], local[:, 2]


def roi_rectangle(points, camera, padding=4):
    pixels, z = project(points, camera)
    low, high = np.floor(pixels.min(0)).astype(int)-padding, np.ceil(pixels.max(0)).astype(int)+padding+1
    width, height = camera['width'], camera['height']
    x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
    x1, y1 = min(width, int(high[0])), min(height, int(high[1]))
    mask = np.zeros((height, width), bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask, {'unclipped_pixel_bbox': [low.tolist(), high.tolist()],
                  'continuous_pixel_bbox': [pixels.min(0).tolist(), pixels.max(0).tolist()],
                  'positive_camera_z_range': [float(z.min()), float(z.max())],
                  'roi_pixels': int(mask.sum()), 'padding_pixels': padding}


def world_hits(buffer, camera, mask):
    selected = np.asarray(mask, bool) & buffer['mask'] & np.isfinite(buffer['depth'])
    y, x = np.nonzero(selected)
    rays = np.column_stack((x+.5, y+.5, np.ones(len(x)))) @ np.linalg.inv(camera['K']).T
    local = rays * (buffer['depth'][selected]/rays[:, 2])[:, None]
    world = local @ camera['c2w'][:3, :3].T+camera['c2w'][:3, 3]
    return selected, world


def json_camera(camera):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in camera.items()}


def srgb8(linear):
    x = np.clip(np.asarray(linear), 0, 1)
    display = np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)
    return np.rint(np.clip(display, 0, 1)*255).astype(np.uint8)


def boundary_mask(mask):
    mask = np.asarray(mask, bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def save_geometry_previews(folder, buffer, depth_limits):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    mask = buffer['mask']; depth = np.zeros(mask.shape, np.uint8)
    lo, hi = depth_limits
    depth[mask] = np.rint(np.clip((buffer['depth'][mask]-lo)/(hi-lo), 0, 1)*254+1).astype(np.uint8)
    normals = np.zeros((*mask.shape, 3), np.uint8)
    normals[mask] = np.rint(np.clip((buffer['normals'][mask]+1)/2, 0, 1)*255).astype(np.uint8)
    silhouette = np.zeros(mask.shape, np.uint8); silhouette[mask] = 110
    silhouette[boundary_mask(mask)] = 255
    for name, array in [('depth', depth), ('normal', normals), ('silhouette', silhouette)]:
        Image.fromarray(array).save(folder/(name+'.png'))


def save_rgb(folder, rgb):
    pixels = srgb8(rgb)
    Image.fromarray(pixels).save(Path(folder)/'rgb.png')
    return pixels


def save_difference(path, first, second, mask, scale, kind='signed'):
    """Fixed-scale diagnostic heatmap; never present as an unamplified image."""
    values = np.asarray(first)-np.asarray(second)
    if values.ndim == 3:
        values = np.linalg.norm(values, axis=2)
    valid = np.asarray(mask, bool) & np.isfinite(values)
    pixels = np.zeros((*valid.shape, 3), np.uint8)
    if kind == 'signed':
        x = np.clip(values[valid]/scale, -1, 1)
        pixels[valid] = np.rint(255*np.column_stack((np.maximum(x, 0), .15*np.abs(x), np.maximum(-x, 0)))).astype(np.uint8)
    else:
        x = np.clip(np.abs(values[valid])/scale, 0, 1)
        pixels[valid] = np.rint(255*np.column_stack((x, .7*x, .05*x))).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def contact_sheet(output, folders, labels, title):
    panel_w, panel_h, label_h = 320, 180, 24
    rows = ('depth', 'normal', 'rgb')
    sheet = Image.new('RGB', (panel_w*len(folders), 44+(panel_h+label_h)*len(rows)), (22, 22, 22))
    draw = ImageDraw.Draw(sheet); draw.text((8, 7), title, fill=(255, 255, 255))
    draw.text((8, 23), 'Half-resolution overview; open individual PNGs for native 640x360.', fill=(210, 210, 210))
    for r, name in enumerate(rows):
        y = 44+r*(panel_h+label_h)
        for c, (folder, label) in enumerate(zip(folders, labels)):
            path = Path(folder)/(name+'.png')
            if not path.is_file():
                continue
            with Image.open(path) as source:
                panel = source.convert('RGB').resize((panel_w, panel_h), Image.Resampling.NEAREST)
            sheet.paste(panel, (c*panel_w, y+label_h))
            draw.text((c*panel_w+5, y+5), label+' / '+name, fill=(235, 235, 235))
    sheet.save(output)
