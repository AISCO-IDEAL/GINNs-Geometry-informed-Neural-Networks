# eschaerer addition
import math

import torch

from GINN.problems.constraints import SampleConstraint, SampleConstraintWithNormals
from GINN.problems.problem_base import ProblemBase
from util.visualization.utils_mesh import (
    get_meshgrid_for_marching_squares,
    get_meshgrid_in_domain,
)

"""
2D drillhole problem: hole boundaries as interfaces (equality constraint),
domain rectangle as envelope (inequality constraint). Constraints are defined
ANALYTICALLY from the drillhole geometry (no point-cloud files).

Registered in util/misc.py:get_problem via problem_str == 'drillhole'.

The geometry is passed directly through the config and normalised to GINN's
roughly [-1, 1] convention in __init__; all sampling happens in the normalised
coordinate system.

Sign convention (GINN SDF): negative inside material, positive outside. Interface
normals point INTO the holes (direction of increasing SDF = outward from material).
"""


def _sd_circle_torch(ts_points: torch.Tensor, ts_center: torch.Tensor, f_radius: float) -> torch.Tensor:
    """Exact Euclidean SDF of a circle; shape-centric sign: negative inside the circle."""
    return torch.norm(ts_points - ts_center, dim=1) - f_radius


def _sd_convex_polygon_torch(
    ts_points: torch.Tensor, ts_vertices: torch.Tensor
) -> torch.Tensor:
    """Exact Euclidean SDF of a convex polygon; shape-centric sign: negative inside the polygon.

    ts_vertices: [M, 2] counter-clockwise.
    ts_points: [N, 2]
    returns: [N]
    """
    ts_edge = torch.roll(ts_vertices, -1, dims=0) - ts_vertices
    ts_pa = ts_points.unsqueeze(1) - ts_vertices.unsqueeze(0)  # [N, M, 2]
    # project pa onto each edge
    ts_t = torch.clamp(
        (ts_pa * ts_edge.unsqueeze(0)).sum(dim=2) / (ts_edge * ts_edge).sum(dim=1),
        min=0.0,
        max=1.0,
    )  # [N, M]
    ts_closest = ts_vertices.unsqueeze(0) + ts_t.unsqueeze(2) * ts_edge.unsqueeze(0)
    ts_dist = torch.norm(ts_points.unsqueeze(1) - ts_closest, dim=2).min(dim=1)[0]
    # cross product sign: inside vs outside
    ts_pa_edge = ts_pa[:, :, 0] * ts_edge[None, :, 1] - ts_pa[:, :, 1] * ts_edge[None, :, 0]
    # inside for either vertex winding (the sign of all cross products agrees)
    ts_inside = (ts_pa_edge >= 0.0).all(dim=1) | (ts_pa_edge <= 0.0).all(dim=1)
    ts_sdf = torch.where(ts_inside, -ts_dist, ts_dist)
    return ts_sdf


def _material_sdf_torch(
    ts_points: torch.Tensor,
    i_shape_code: int,
    ts_center: torch.Tensor,
    f_radius: float,
    ts_vertices: torch.Tensor,
) -> torch.Tensor:
    """Material SDF: positive inside the hole (no material), negative in solid material."""
    ts_local = ts_points - ts_center
    if i_shape_code == 1:  # circle
        return -_sd_circle_torch(ts_local, torch.zeros(2, device=ts_points.device), f_radius)
    return -_sd_convex_polygon_torch(ts_local, ts_vertices)


class ProblemDrillhole(ProblemBase):

    def __init__(self, nx,
                 domain_width, domain_height,
                 holes,  # list of dicts: shape_code, center [x,y], radius, aspect_ratio, vertices
                 n_points_domain,
                 n_points_envelope,
                 n_points_interfaces,
                 area_fraction_min,
                 area_fraction_max=1.0,
                 plot_2d_resolution=100,
                 domain_margin=0.0,  # mm; the SDF must not cross the inner box inset by this margin
                 **kwargs) -> None:
        super().__init__(nx=nx)

        device = torch.get_default_device()

        # Domain in mm
        self.domain_width = float(domain_width)
        self.domain_height = float(domain_height)
        f_scale = max(self.domain_width, self.domain_height) / 2.0
        ts_center_norm = torch.tensor(
            [self.domain_width / 2.0 / f_scale, self.domain_height / 2.0 / f_scale],
            dtype=torch.float32,
            device=device,
        )

        # Normalised bounds: [[-w/2s, w/2s], [-h/2s, h/2s]]
        self.bounds = torch.tensor(
            [
                [-self.domain_width / 2.0 / f_scale, self.domain_width / 2.0 / f_scale],
                [-self.domain_height / 2.0 / f_scale, self.domain_height / 2.0 / f_scale],
            ],
            dtype=torch.float32,
            device=device,
        )
        f_margin_norm = float(domain_margin) / f_scale
        self.envelope = self.bounds.clone()
        self.envelope[:, 0] += f_margin_norm
        self.envelope[:, 1] -= f_margin_norm
        self.f_area_min = float(area_fraction_min)
        self.f_area_max = float(area_fraction_max)
        self.n_points_domain = n_points_domain
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces

        # Convert holes to normalised coordinates
        self.lst_holes_norm = []
        for hole in holes:
            arr_center = torch.tensor(hole["center"], dtype=torch.float32, device=device)
            center_norm = (arr_center / f_scale) - ts_center_norm
            radius_norm = float(hole["radius"]) / f_scale
            vertices_norm = torch.empty((0, 2), dtype=torch.float32, device=device)
            if hole["shape_code"] != 1:  # polygon
                vertices_raw = torch.tensor(hole["vertices"], dtype=torch.float32, device=device)
                vertices_norm = (vertices_raw / f_scale) - ts_center_norm
            self.lst_holes_norm.append({
                "shape_code": hole["shape_code"],
                "center": center_norm,
                "radius": radius_norm,
                "aspect_ratio": float(hole.get("aspect_ratio", 1.0)),
                "vertices": vertices_norm,
            })

        # Constraints built from analytic SDFs (no files)
        interface_constr = _AnalyticInterfaceConstraint(self)
        envelope_constr = _AnalyticEnvelopeConstraint(self)
        inside_constr = _AnalyticInsideConstraint(self)

        self.constr_pts_dict = {}

        self._envelope_constr = [envelope_constr]
        self._interface_constraints = [interface_constr]
        self._obstacle_constraints = []
        self._inside_envelope = inside_constr
        self._domain = inside_constr

        # Grid for contour/meshing
        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.bounds.cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32, device=device)
        self.X0, self.X1, xs = get_meshgrid_in_domain(
            self.bounds.cpu(), plot_2d_resolution, plot_2d_resolution
        )
        self.xs = torch.tensor(xs, dtype=torch.float32, device=device)

    def is_inside_envelope(self, a: torch.Tensor):
        """Get mask for points which are inside the envelope (domain)."""
        return (a[:, 0] >= self.envelope[0, 0]) & (a[:, 0] <= self.envelope[0, 1]) & \
            (a[:, 1] >= self.envelope[1, 0]) & (a[:, 1] <= self.envelope[1, 1])


class _AnalyticInterfaceConstraint:
    """Interface constraint: SDF = level_set on hole boundaries (equality)."""

    def __init__(self, problem: ProblemDrillhole):
        self.problem = problem

    def get_sampled_points(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return N points on hole boundaries with into-hole unit normals, in normalised coords."""
        import torch
        device = self.problem.bounds.device
        rng = torch.Generator(device=device)
        rng = torch.Generator(device=device)
        rng.manual_seed(0)  # make deterministic for now

        lst_pts, lst_normals = [], []
        lst_lengths = []
        for hole in self.problem.lst_holes_norm:
            if hole["shape_code"] == 1:  # circle
                lst_lengths.append(2.0 * math.pi * hole["radius"])
            else:
                arr_edge = torch.roll(hole["vertices"], -1, dims=0) - hole["vertices"]
                lst_lengths.append(torch.norm(arr_edge, dim=1).sum().item())
        arr_weights = torch.tensor(lst_lengths, device=device)
        arr_weights = arr_weights / arr_weights.sum()
        lst_counts = (arr_weights * N).floor().to(int)
        remainder = N - lst_counts.sum().item()
        if remainder > 0:
            lst_counts[torch.argmax(arr_weights)] += remainder

        for hole, i_count in zip(self.problem.lst_holes_norm, lst_counts, strict=True):
            i_count = int(i_count)
            if i_count == 0:
                continue
            if hole["shape_code"] == 1:  # circle
                arr_angle = torch.linspace(0.0, 2.0 * math.pi, i_count + 1, device=device)[:-1]
                arr_dir = torch.stack([arr_angle.cos(), arr_angle.sin()], dim=1)
                pts = hole["center"] + hole["radius"] * arr_dir
                normals = -arr_dir  # into the hole, like the polygon normals and the SDF gradient
            else:
                arr_edge = torch.roll(hole["vertices"], -1, dims=0) - hole["vertices"]
                arr_edge_len = torch.norm(arr_edge, dim=1)
                arr_edge_idx = torch.multinomial(arr_edge_len, i_count, replacement=True)
                arr_t = torch.rand(i_count, device=device)
                pts = hole["vertices"][arr_edge_idx] + arr_t.unsqueeze(1) * arr_edge[arr_edge_idx] + hole["center"]
                # CCW polygon: outward normal = (-dy, dx) / |edge|
                edge_norm = arr_edge_len[arr_edge_idx]
                normals = torch.stack([-arr_edge[arr_edge_idx, 1], arr_edge[arr_edge_idx, 0]], dim=1) / edge_norm.unsqueeze(1)
            lst_pts.append(pts)
            lst_normals.append(normals)

        return torch.cat(lst_pts), torch.cat(lst_normals)


class _AnalyticEnvelopeConstraint:
    """Envelope constraint: SDF >= level_set outside/on the domain rectangle (inequality)."""

    def __init__(self, problem: ProblemDrillhole):
        self.problem = problem

    def get_sampled_points(self, N: int) -> torch.Tensor:
        import torch
        device = self.problem.envelope.device
        rng = torch.Generator(device=device)
        rng.manual_seed(0)
        i_on = N // 2
        i_out = N - i_on

        # on the rectangle: pick a side proportional to length, then uniform
        sides = torch.tensor([
            [self.problem.envelope[0, 0], self.problem.envelope[1, 0]],
            [self.problem.envelope[0, 1], self.problem.envelope[1, 0]],
            [self.problem.envelope[0, 1], self.problem.envelope[1, 1]],
            [self.problem.envelope[0, 0], self.problem.envelope[1, 1]],
        ], device=device)  # corners
        side_lens = torch.tensor([
            (sides[0, 0] - sides[1, 0]).abs(),
            (sides[1, 1] - sides[0, 1]).abs(),
            (sides[2, 0] - sides[3, 0]).abs(),
            (sides[3, 1] - sides[2, 1]).abs(),
        ], device=device)
        side_weights = side_lens / side_lens.sum()
        side_idx = torch.multinomial(side_weights, i_on, replacement=True)
        t = torch.rand(i_on, device=device)

        def lerp(c0, c1):
            return c0 + t.unsqueeze(1) * (c1 - c0)
        corner_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        arr_on = torch.empty((i_on, 2), device=device)
        for i, idx in enumerate(side_idx):
            c0_idx, c1_idx = corner_pairs[idx.item()]
            arr_on[i] = lerp(sides[c0_idx], sides[c1_idx])[i]

        # outside: rejection sampling in inflated rectangle
        margin = 0.1
        lst_out = []
        i_rem = i_out
        while i_rem > 0:
            batch = 2 * i_rem
            cand = torch.rand(batch, 2, device=device) * (
                1.0 + 2.0 * margin
            ) - (0.5 + margin)
            cand = cand * torch.tensor([self.problem.envelope[0, 1] - self.problem.envelope[0, 0],
                                        self.problem.envelope[1, 1] - self.problem.envelope[1, 0]], device=device) * 0.5
            cand = cand + torch.tensor([self.problem.envelope[:, 0].mean(),
                                        self.problem.envelope[:, 1].mean()], device=device)
            b_out = self.problem.is_inside_envelope(cand).logical_not()
            lst_out.append(cand[b_out])
            i_rem -= int(b_out.sum().item())
        arr_out = torch.cat(lst_out)[:i_out]
        return torch.cat([arr_on, arr_out])


class _AnalyticInsideConstraint:
    """Domain interior points (outside holes) for eikonal loss."""

    def __init__(self, problem: ProblemDrillhole):
        self.problem = problem

    def get_sampled_points(self, N: int) -> torch.Tensor:
        import torch
        device = self.problem.bounds.device
        rng = torch.Generator(device=device)
        rng.manual_seed(0)
        lst_in = []
        i_rem = N
        while i_rem > 0:
            batch = 2 * i_rem
            cand = torch.rand(batch, 2, device=device)
            # map to bounds
            cand = cand * (self.problem.envelope[:, 1] - self.problem.envelope[:, 0]) + self.problem.envelope[:, 0]
            # reject if inside any hole
            sdfs = torch.stack([
                _material_sdf_torch(
                    cand,
                    h["shape_code"],
                    h["center"],
                    h["radius"],
                    h["vertices"],
                )
                for h in self.problem.lst_holes_norm
            ])
            # material SDF < 0 means IN material (solid, not in hole)
            b_in = (sdfs < 0.0).all(dim=0)
            lst_in.append(cand[b_in])
            i_rem -= int(b_in.sum().item())
        return torch.cat(lst_in)[:N]
