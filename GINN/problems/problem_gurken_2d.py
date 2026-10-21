# gurken addition
"""2D problem from plain geometry in mm: envelope, interfaces and excluded shapes.

Written for the experiments of the gurken project other than the drillhole one (the cantilever
beam); registered in util/misc.py:get_problem via problem_str == 'gurken_2d'. The geometry comes
through the config (``problem`` section, mm, lower-left corner of the domain at the origin):

- ``domain_width``, ``domain_height``, ``domain_margin`` (the envelope is the domain inset by it);
- ``interfaces``: where SDF = level_set, with the target normal (unit, into the excluded region,
  the direction of increasing SDF):
  ``{"kind": "circle", "center": [x, y], "radius": r}`` (normal into the circle) or
  ``{"kind": "segment", "start": [x, y], "end": [x, y], "normal": [nx, ny]}``;
- ``excluded``: regions without material, kept out of the interior samples:
  ``{"kind": "circle", "center", "radius"}`` or ``{"kind": "polygon", "vertices": [[x, y], ...]}``
  (any simple polygon, absolute coordinates).

Normalised like ProblemDrillhole: ``x_norm = (x_mm - domain centre) / (max(W, H) / 2)``. Without
interfaces the interface sample is a single point far outside the domain, because GINN's surface
helper needs one; lambda_if and lambda_if_normal must then be 0.

Sign convention (GINN SDF): negative inside material, positive outside.
"""

import math

import torch

from GINN.problems.problem_base import ProblemBase
from GINN.problems.problem_drillhole import _AnalyticEnvelopeConstraint
from util.visualization.utils_mesh import (
    get_meshgrid_for_marching_squares,
    get_meshgrid_in_domain,
)

F_FAR_NORM = 10.0  # normalised coordinate of the stand-in interface point


def _inside_polygon(ts_points: torch.Tensor, ts_vertices: torch.Tensor) -> torch.Tensor:
    """Even-odd rule: True for points [N, 2] inside the simple polygon [M, 2] (any winding)."""
    ts_next = torch.roll(ts_vertices, -1, dims=0)
    ts_py = ts_points[:, None, 1]
    ts_straddles = (ts_vertices[None, :, 1] > ts_py) != (ts_next[None, :, 1] > ts_py)
    ts_dy = ts_next[:, 1] - ts_vertices[:, 1]
    ts_dy = torch.where(ts_dy == 0.0, torch.full_like(ts_dy, 1e-30), ts_dy)
    ts_x_cross = ts_vertices[None, :, 0] + (ts_py - ts_vertices[None, :, 1]) * (
        (ts_next[:, 0] - ts_vertices[:, 0]) / ts_dy
    )[None, :]
    return (ts_straddles & (ts_points[:, None, 0] < ts_x_cross)).sum(dim=1) % 2 == 1


class ProblemGurken2d(ProblemBase):

    def __init__(self, nx,
                 domain_width, domain_height,
                 n_points_domain,
                 n_points_envelope,
                 n_points_interfaces,
                 interfaces=(),
                 excluded=(),
                 plot_2d_resolution=100,
                 domain_margin=0.0,
                 **kwargs) -> None:
        super().__init__(nx=nx)
        device = torch.get_default_device()
        self.domain_width = float(domain_width)
        self.domain_height = float(domain_height)
        self.f_scale = max(self.domain_width, self.domain_height) / 2.0
        self.ts_center_mm = torch.tensor(
            [self.domain_width / 2.0, self.domain_height / 2.0], dtype=torch.float32, device=device
        )
        f_half_w, f_half_h = self.domain_width / 2.0 / self.f_scale, self.domain_height / 2.0 / self.f_scale
        self.bounds = torch.tensor(
            [[-f_half_w, f_half_w], [-f_half_h, f_half_h]], dtype=torch.float32, device=device
        )
        f_margin_norm = float(domain_margin) / self.f_scale
        self.envelope = self.bounds.clone()
        self.envelope[:, 0] += f_margin_norm
        self.envelope[:, 1] -= f_margin_norm
        self.n_points_domain = n_points_domain
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces

        self.lst_interfaces = [self._normalise(d) for d in interfaces]
        self.lst_excluded = [self._normalise(d) for d in excluded]

        inside_constr = _InsideConstraint(self)
        self.constr_pts_dict = {}
        self._envelope_constr = [_AnalyticEnvelopeConstraint(self)]
        self._interface_constraints = [_InterfaceConstraint(self)]
        self._obstacle_constraints = []
        self._inside_envelope = inside_constr
        self._domain = inside_constr

        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.bounds.cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32, device=device)
        self.X0, self.X1, xs = get_meshgrid_in_domain(
            self.bounds.cpu(), plot_2d_resolution, plot_2d_resolution
        )
        self.xs = torch.tensor(xs, dtype=torch.float32, device=device)

    def _point(self, lst_xy) -> torch.Tensor:
        ts_xy = torch.tensor(lst_xy, dtype=torch.float32, device=self.bounds.device)
        return (ts_xy - self.ts_center_mm) / self.f_scale

    def _normalise(self, dict_shape: dict) -> dict:
        """A shape of the config in normalised coordinates."""
        str_kind = dict_shape["kind"]
        if str_kind == "circle":
            return {"kind": str_kind, "center": self._point(dict_shape["center"]),
                    "radius": float(dict_shape["radius"]) / self.f_scale}
        if str_kind == "segment":
            ts_normal = torch.tensor(dict_shape["normal"], dtype=torch.float32, device=self.bounds.device)
            return {"kind": str_kind, "start": self._point(dict_shape["start"]),
                    "end": self._point(dict_shape["end"]), "normal": ts_normal / ts_normal.norm()}
        if str_kind == "polygon":
            return {"kind": str_kind, "vertices": self._point(dict_shape["vertices"])}
        raise ValueError(f"Unknown shape kind {str_kind!r}")

    def is_inside_envelope(self, a: torch.Tensor):
        """Get mask for points which are inside the envelope (domain)."""
        return (a[:, 0] >= self.envelope[0, 0]) & (a[:, 0] <= self.envelope[0, 1]) & \
            (a[:, 1] >= self.envelope[1, 0]) & (a[:, 1] <= self.envelope[1, 1])

    def is_excluded(self, ts_points: torch.Tensor) -> torch.Tensor:
        """True for points [N, 2] (normalised) inside any excluded shape."""
        b_out = torch.zeros(len(ts_points), dtype=torch.bool, device=ts_points.device)
        for shape in self.lst_excluded:
            if shape["kind"] == "circle":
                b_out |= (ts_points - shape["center"]).norm(dim=1) < shape["radius"]
            else:
                b_out |= _inside_polygon(ts_points, shape["vertices"])
        return b_out


class _InterfaceConstraint:
    """SDF = level_set on the interfaces, target normals into the excluded regions."""

    def __init__(self, problem: ProblemGurken2d):
        self.problem = problem

    def get_sampled_points(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.problem.bounds.device
        lst_shapes = self.problem.lst_interfaces
        if not lst_shapes:
            ts_far = torch.full((1, 2), F_FAR_NORM, device=device)
            return ts_far, torch.tensor([[1.0, 0.0]], device=device)
        ts_length = torch.tensor([
            2.0 * math.pi * s["radius"] if s["kind"] == "circle" else (s["end"] - s["start"]).norm().item()
            for s in lst_shapes
        ], device=device)
        ts_counts = (ts_length / ts_length.sum() * N).floor().to(int)
        ts_counts[torch.argmax(ts_length)] += N - int(ts_counts.sum())
        lst_pts, lst_normals = [], []
        for shape, i_count in zip(lst_shapes, ts_counts.tolist(), strict=True):
            if i_count == 0:
                continue
            if shape["kind"] == "circle":
                ts_angle = torch.linspace(0.0, 2.0 * math.pi, i_count + 1, device=device)[:-1]
                ts_dir = torch.stack([ts_angle.cos(), ts_angle.sin()], dim=1)
                lst_pts.append(shape["center"] + shape["radius"] * ts_dir)
                lst_normals.append(-ts_dir)  # into the circle
            else:
                ts_t = torch.rand(i_count, 1, device=device)
                lst_pts.append(shape["start"] + ts_t * (shape["end"] - shape["start"]))
                lst_normals.append(shape["normal"].expand(i_count, 2))
        return torch.cat(lst_pts), torch.cat(lst_normals)


class _InsideConstraint:
    """Points in the envelope outside every excluded shape (eikonal loss)."""

    def __init__(self, problem: ProblemGurken2d):
        self.problem = problem

    def get_sampled_points(self, N: int) -> torch.Tensor:
        device = self.problem.bounds.device
        envelope = self.problem.envelope
        lst_in, i_rem = [], N
        while i_rem > 0:
            cand = torch.rand(2 * i_rem, 2, device=device) * (envelope[:, 1] - envelope[:, 0]) + envelope[:, 0]
            cand = cand[~self.problem.is_excluded(cand)]
            lst_in.append(cand)
            i_rem -= len(cand)
        return torch.cat(lst_in)[:N]
