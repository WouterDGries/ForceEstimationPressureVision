"""Depth-to-color alignment in numpy, restricted to a region of interest of
the color image. Replaces running rs.align over the whole frame: the live
preview only needs depth around the fingertip, and the offline extractor
only needs it inside the crop, so aligning the full 1920x1080 frame every
frame is wasted work.

Works on plain dicts (see intrinsics_to_dict / extrinsics_to_dict) so the
offline extractor can align recorded frames without a camera attached. The
projection math mirrors librealsense's rs2_project_point_to_pixel /
rs2_deproject_pixel_to_point, and each depth pixel is splatted over the
color pixels its footprint covers (like rs.align), keeping the nearest
depth where several land on the same color pixel.
"""

import numpy as np

_SUPPORTED_MODELS = ("none", "brown_conrady", "inverse_brown_conrady", "modified_brown_conrady")


def intrinsics_to_dict(intrinsics):
    """Input: pyrealsense2 intrinsics. Returns a JSON-serializable dict."""
    return {
        "width": intrinsics.width, "height": intrinsics.height,
        "fx": intrinsics.fx, "fy": intrinsics.fy, "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
        "model": str(intrinsics.model).split(".")[-1],
        "coeffs": list(intrinsics.coeffs),
    }


def extrinsics_to_dict(extrinsics):
    """Input: pyrealsense2 extrinsics. Returns a JSON-serializable dict with
    a row-major 3x3 rotation (librealsense stores it column-major) and a
    translation in meters, so that p_to = R @ p_from + t.
    """
    rotation = np.array(extrinsics.rotation, dtype=np.float64).reshape(3, 3, order="F")
    return {"rotation": rotation.tolist(), "translation": list(extrinsics.translation)}


def invert_extrinsics(extrinsics):
    """Input: extrinsics dict. Returns the inverse transform as a dict."""
    rotation = np.array(extrinsics["rotation"])
    translation = np.array(extrinsics["translation"])
    return {"rotation": rotation.T.tolist(), "translation": (-rotation.T @ translation).tolist()}


def _check_model(intrinsics):
    if intrinsics["model"] not in _SUPPORTED_MODELS and any(intrinsics["coeffs"]):
        raise ValueError(f"Unsupported distortion model {intrinsics['model']} with non-zero coefficients")


def project(intrinsics, x, y, z):
    """Input: intrinsics dict, 3D points (arrays, meters). Returns pixel (u, v) float arrays."""
    xn, yn = x / z, y / z
    c = intrinsics["coeffs"]
    model = intrinsics["model"]
    if any(c) and model in ("modified_brown_conrady", "inverse_brown_conrady", "brown_conrady"):
        r2 = xn * xn + yn * yn
        f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
        if model == "brown_conrady":
            dx = xn * f + 2 * c[2] * xn * yn + c[3] * (r2 + 2 * xn * xn)
            dy = yn * f + 2 * c[3] * xn * yn + c[2] * (r2 + 2 * yn * yn)
        else:
            xf, yf = xn * f, yn * f
            dx = xf + 2 * c[2] * xf * yf + c[3] * (r2 + 2 * xf * xf)
            dy = yf + 2 * c[3] * xf * yf + c[2] * (r2 + 2 * yf * yf)
        xn, yn = dx, dy
    return xn * intrinsics["fx"] + intrinsics["ppx"], yn * intrinsics["fy"] + intrinsics["ppy"]


def deproject(intrinsics, u, v, z):
    """Input: intrinsics dict, pixel coords (arrays), depth in meters.
    Returns 3D points (x, y, z) in meters.
    """
    xn = (u - intrinsics["ppx"]) / intrinsics["fx"]
    yn = (v - intrinsics["ppy"]) / intrinsics["fy"]
    c = intrinsics["coeffs"]
    model = intrinsics["model"]
    if any(c) and model == "inverse_brown_conrady":
        r2 = xn * xn + yn * yn
        f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
        ux = xn * f + 2 * c[2] * xn * yn + c[3] * (r2 + 2 * xn * xn)
        uy = yn * f + 2 * c[3] * xn * yn + c[2] * (r2 + 2 * yn * yn)
        xn, yn = ux, uy
    elif any(c) and model == "brown_conrady":
        x0, y0 = xn, yn
        for _ in range(10):
            r2 = xn * xn + yn * yn
            icdist = 1 / (1 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2)
            delta_x = 2 * c[2] * xn * yn + c[3] * (r2 + 2 * xn * xn)
            delta_y = 2 * c[3] * xn * yn + c[2] * (r2 + 2 * yn * yn)
            xn = (x0 - delta_x) * icdist
            yn = (y0 - delta_y) * icdist
    return xn * z, yn * z, z


def _transform(extrinsics, x, y, z):
    r = extrinsics["rotation"]
    t = extrinsics["translation"]
    return (r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0],
            r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1],
            r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2])


class DepthAligner:
    """Aligns raw depth frames onto a region of the color image.

    Build once per session from the stream calibration (dicts as stored in
    meta_raw.json), then call align_roi(depth_u16, box) per frame.
    """

    def __init__(self, depth_intrinsics, color_intrinsics, depth_to_color, depth_scale_m,
                 min_depth_m=0.1, max_depth_m=1.0):
        _check_model(depth_intrinsics)
        _check_model(color_intrinsics)
        self.depth_intrinsics = depth_intrinsics
        self.color_intrinsics = color_intrinsics
        self.depth_to_color = depth_to_color
        self.color_to_depth = invert_extrinsics(depth_to_color)
        self.depth_scale_m = depth_scale_m
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def _depth_box_for_color_box(self, box):
        """Returns the depth-image box that can map into the color box, by
        projecting the color box corners at the near and far depth limits.
        """
        x0, y0, x1, y1 = box
        corners_u = np.array([x0, x1, x0, x1, x0, x1, x0, x1], dtype=np.float64)
        corners_v = np.array([y0, y0, y1, y1, y0, y0, y1, y1], dtype=np.float64)
        corners_z = np.array([self.min_depth_m] * 4 + [self.max_depth_m] * 4)
        points = deproject(self.color_intrinsics, corners_u, corners_v, corners_z)
        u, v = project(self.depth_intrinsics, *_transform(self.color_to_depth, *points))
        width, height = self.depth_intrinsics["width"], self.depth_intrinsics["height"]
        dx0 = int(max(np.floor(u.min()) - 2, 0))
        dy0 = int(max(np.floor(v.min()) - 2, 0))
        dx1 = int(min(np.ceil(u.max()) + 3, width))
        dy1 = int(min(np.ceil(v.max()) + 3, height))
        return dx0, dy0, dx1, dy1

    def align_roi(self, depth_u16, box):
        """Input: raw (unaligned) uint16 depth frame and a color-image box
        (x0, y0, x1, y1), end-exclusive, may extend past the image. Returns
        a (y1-y0, x1-x0) uint16 array of depth aligned to those color
        pixels, 0 where there is no reading.
        """
        x0, y0, x1, y1 = box
        out_w, out_h = x1 - x0, y1 - y0
        dx0, dy0, dx1, dy1 = self._depth_box_for_color_box(box)
        out = np.full(out_h * out_w, np.iinfo(np.uint16).max, dtype=np.uint16)
        if dx1 <= dx0 or dy1 <= dy0:
            return np.zeros((out_h, out_w), dtype=np.uint16)

        region = depth_u16[dy0:dy1, dx0:dx1]
        rows, cols = np.nonzero(region)
        raw = region[rows, cols]
        z = raw.astype(np.float64) * self.depth_scale_m
        du = cols.astype(np.float64) + dx0
        dv = rows.astype(np.float64) + dy0

        # Each depth pixel covers [u-0.5, u+0.5] x [v-0.5, v+0.5]; map both
        # corners into color and fill every color pixel in between.
        u_a, v_a = project(self.color_intrinsics, *_transform(
            self.depth_to_color, *deproject(self.depth_intrinsics, du - 0.5, dv - 0.5, z)))
        u_b, v_b = project(self.color_intrinsics, *_transform(
            self.depth_to_color, *deproject(self.depth_intrinsics, du + 0.5, dv + 0.5, z)))
        cu0 = np.floor(np.minimum(u_a, u_b) + 0.5).astype(np.int64) - x0
        cv0 = np.floor(np.minimum(v_a, v_b) + 0.5).astype(np.int64) - y0
        cu1 = np.floor(np.maximum(u_a, u_b) + 0.5).astype(np.int64) - x0
        cv1 = np.floor(np.maximum(v_a, v_b) + 0.5).astype(np.int64) - y0

        max_span = int(max(np.max(cu1 - cu0, initial=0), np.max(cv1 - cv0, initial=0)))
        for oy in range(max_span + 1):
            for ox in range(max_span + 1):
                tu, tv = cu0 + ox, cv0 + oy
                keep = (tu <= cu1) & (tv <= cv1) & (tu >= 0) & (tv >= 0) & (tu < out_w) & (tv < out_h)
                if np.any(keep):
                    np.minimum.at(out, tv[keep] * out_w + tu[keep], raw[keep])

        out[out == np.iinfo(np.uint16).max] = 0
        return out.reshape(out_h, out_w)
