"""Vue 3D : caméra orbitale (numpy) et rendu GPU (Taichi) dans l'image du solveur.

    cam = Camera.fit(extent)                  # cadre la boîte (Lx, Ly, Lz)
    cam.orbit(dx_px, dy_px) ; cam.pan(...) ; cam.zoom(1.25)
    img = solver.render3d(cam, size=(w, h), clip=(2, 0.3, True))   # (h, w, 3) u8

Rendu, entièrement sur le GPU (seule l'image finale est copiée vers le CPU, comme la vue 2D) :
  1. fond, un rayon par pixel : obstacles tracés sur la distance signée du solveur (sphere tracing, normale =
     gradient), sinon face du fond de la boîte colorée par type de paroi (mur / entrée / sortie) ;
  2. particules (fluide puis solide) en sphères : passe de profondeur (atomic_min dans un tampon f32), puis passe
     de couleur (ombrage de Lambert, normale tirée de la position dans le disque) si la profondeur est la bonne.
Plan de coupe : les particules au-delà d'un plan x, y ou z = cte sont masquées (voir l'intérieur).
Axes : x à droite, y vertical (gravité), z profondeur. Pixel (ligne 0 en haut, colonne 0 à gauche).
"""
import math

import numpy as np
import taichi as ti

from Solver.boundary import FRAMES, INLET, OUTLET
from ui.kernels import particle_color, rotor_sdf, sdf_at

WORLD_UP = np.array([0.0, 1.0, 0.0])


class Camera:
    """Caméra orbitale : cible, lacet / tangage (degrés), distance, angle de vue vertical (degrés)."""

    def __init__(self, target=(0.5, 0.5, 0.5), yaw: float = 30.0, pitch: float = 20.0, dist: float = 2.5,
                 fov: float = 40.0):
        self.target = np.asarray(target, np.float64)
        self.yaw, self.pitch, self.dist, self.fov = float(yaw), float(pitch), float(dist), float(fov)

    @classmethod
    def fit(cls, extent, yaw: float = 30.0, pitch: float = 20.0, fov: float = 40.0) -> "Camera":
        """Camera framing a box [0, extent].

        **Inputs**

        - `extent` : (3,) float box size (m) ; `yaw`, `pitch`, `fov` : float degrees

        **Outputs**

        - Camera
        """
        ext = np.asarray(extent, np.float64)
        radius = 0.5 * float(np.linalg.norm(ext))
        dist = radius / math.sin(math.radians(fov) * 0.5) * 1.05
        return cls(0.5 * ext, yaw, pitch, dist, fov)

    def basis(self) -> tuple:
        """Eye position and orthonormal frame.

        **Outputs**

        - (eye (3,), forward (3,), right (3,), up (3,))
        """
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        eye = self.target + self.dist * np.array([cp * sy, sp, cp * cy])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, WORLD_UP)
        right /= max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def tan_half(self) -> float:
        """tan(fov / 2) (float)."""
        return math.tan(math.radians(self.fov) * 0.5)

    def orbit(self, dx_px: float, dy_px: float, speed: float = 0.4) -> None:
        """Rotate around the target (mouse drag in pixels)."""
        self.yaw -= dx_px * speed
        self.pitch = float(np.clip(self.pitch + dy_px * speed, -89.0, 89.0))

    def pan(self, dx_px: float, dy_px: float, h: int) -> None:
        """Move the target in the view plane (mouse drag in pixels, h = image height in pixels)."""
        _, _, right, up = self.basis()
        k = 2.0 * self.dist * self.tan_half() / max(h, 1)
        self.target = self.target - right * dx_px * k + up * dy_px * k

    def zoom(self, factor: float) -> None:
        """Move closer (factor > 1) or away (factor < 1)."""
        self.dist = float(np.clip(self.dist / factor, 1e-3, 1e4))

    def project(self, pts, w: int, h: int) -> tuple:
        """Project world points to pixels.

        **Inputs**

        - `pts` : np.ndarray (N, 3) ; `w`, `h` : int image size

        **Outputs**

        - (px (N, 2) float pixel coordinates, depth (N,) float along the view axis ; <= 0 : behind the camera)
        """
        eye, fwd, right, up = self.basis()
        v = np.asarray(pts, np.float64).reshape(-1, 3) - eye
        z = v @ fwd
        th = self.tan_half()
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        sx = (v @ right) / (zs * th * (w / h))
        sy = (v @ up) / (zs * th)
        return np.stack([(sx + 1.0) * 0.5 * w, (1.0 - sy) * 0.5 * h], axis=1), z

    def ray(self, px: float, py: float, w: int, h: int) -> tuple:
        """Ray through a pixel.

        **Inputs**

        - `px`, `py` : float pixel position ; `w`, `h` : int image size

        **Outputs**

        - (origin (3,), unit direction (3,))
        """
        eye, fwd, right, up = self.basis()
        th = self.tan_half()
        d = fwd + (2.0 * px / w - 1.0) * th * (w / h) * right + (1.0 - 2.0 * py / h) * th * up
        return eye, d / np.linalg.norm(d)


def ray_box(o, d, extent) -> tuple:
    """Ray / box [0, extent] intersection (numpy, for picking).

    **Inputs**

    - `o`, `d` : (3,) ray origin and direction ; `extent` : (3,) box size

    **Outputs**

    - (t_near, t_far) floats ; t_near > t_far : no hit
    """
    ext = np.asarray(extent, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0, t1 = (0.0 - o) * inv, (ext - o) * inv
    return float(np.max(np.minimum(t0, t1))), float(np.min(np.maximum(t0, t1)))


@ti.func
def _wall_color(t):
    """Color of a wall face by BC type."""
    c = ti.Vector([0.30, 0.31, 0.36])
    if t == INLET:
        c = ti.Vector([0.20, 0.62, 0.30])
    elif t == OUTLET:
        c = ti.Vector([0.70, 0.20, 0.20])
    return c


@ti.data_oriented
class View3D:
    def __init__(self, fb, res: int):
        """Allocate the depth buffer on the solver's field builder.

        **Inputs**

        - `fb` : ti.FieldsBuilder ; `res` : int maximal image side (px)
        """
        self.res = res
        self.zbuf = ti.field(ti.f32)
        self.odepth = ti.field(ti.f32)                   # obstacles translucides : premier contact (distance)
        self.ocol = ti.Vector.field(3, ti.f32)           # et sa couleur ombrée
        fb.dense(ti.ij, (res, res)).place(self.zbuf, self.odepth, self.ocol)
        self.rc0 = ti.Vector.field(3, ti.f32)            # centre de rotor factice (scène sans rotor)
        fb.place(self.rc0)

    def draw(self, s, cam: Camera, w: int, h: int, mode: int, signed: int, inv_smax: float, clip=None,
             prims=None, obs_alpha: float = 1.0) -> None:
        """Render a solver state into s.img[:h, :w].

        **Inputs**

        - `s` : ui.solver.Solver (3D) ; `cam` : Camera ; `w`, `h` : int image size
        - `mode`, `signed`, `inv_smax` : fluid colors (see Solver._colors)
        - `clip` : (axis, position m, keep_below bool) | None ; `prims` : unused (overlays are drawn by the UI)
        - `obs_alpha` : float opacity of the fixed obstacles (1 opaque ; < 1 : translucent, the fluid inside a
          closed casing shows through ; the rotor stays opaque)
        """
        eye, fwd, right, up = cam.basis()
        th = cam.tan_half()
        ext = np.array(s.shape, np.float64) * s.dx
        ca, cpos, ckeep = (-1, 0.0, 1) if clip is None else (int(clip[0]), float(clip[1]), int(bool(clip[2])))
        cam_v = [ti.Vector(list(map(float, v))) for v in (eye, fwd, right, up)]
        has_rot = int(s.rotor is not None)
        rsdf = s.mac.rsdf if has_rot else s.sdf
        rc = s.mac.rc if has_rot else self.rc0
        self._background(s.img, w, h, *cam_v, th, s.sdf, s.use_sdf, s.wall_type, s.p["bound"], s.dx,
                         ti.Vector(list(map(float, ext))), rsdf, rc, has_rot, s.rot_axis, s.rot_omega * s.t,
                         ca, cpos, ckeep, int(obs_alpha < 0.999))
        radius = 0.5 * s.p_spacing * 1.15                 # sphères légèrement jointives
        base = ti.Vector([0.35, 0.65, 1.0])
        if s.has_fluid:
            self._depth(s.x_f, s.alive, 1, w, h, *cam_v, th, radius, ca, cpos, ckeep)
        if s.has_solid:
            self._depth(s.x_s, s.broken_s, 0, w, h, *cam_v, th, radius, ca, cpos, ckeep)
        if s.has_fluid:
            self._shade_fluid(s.img, s.x_f, s.alive, s.sc_f, mode, signed, inv_smax, base, w, h, *cam_v, th, radius,
                              ca, cpos, ckeep)
        if s.has_solid:
            self._shade_solid(s.img, s.x_s, s.col_s, w, h, *cam_v, th, radius, ca, cpos, ckeep)
        if obs_alpha < 0.999 and s.use_sdf:
            self._composite(s.img, w, h, float(obs_alpha))

    @ti.kernel
    def _composite(self, img: ti.template(), w: int, h: int, alpha: float):
        """Blend the translucent obstacle over whatever lies behind it (particles, rotor, far walls)."""
        for row, col in ti.ndrange(h, w):
            if self.odepth[row, col] < self.zbuf[row, col]:
                c = img[row, col].cast(ti.f32) / 255.0
                c = c * (1.0 - alpha) + self.ocol[row, col] * alpha
                img[row, col] = ti.cast(ti.math.clamp(c, 0.0, 1.0) * 255, ti.u8)

    # ------------------------------------------------------------ 1. fond : boîte, parois, obstacles
    @ti.kernel
    def _background(self, img: ti.template(), w: int, h: int, eye: ti.types.vector(3, ti.f32),
                    fwd: ti.types.vector(3, ti.f32), right: ti.types.vector(3, ti.f32),
                    up: ti.types.vector(3, ti.f32), th: float, sdf: ti.template(), use_sdf: int,
                    wall_type: ti.template(), bound: int, dx: float, ext: ti.types.vector(3, ti.f32),
                    rsdf: ti.template(), rc: ti.template(), has_rot: int, axis: ti.types.vector(3, ti.f32),
                    theta: float, ca: int, cpos: float, ckeep: int, translucent: int):
        """Background pass: obstacles and rotor at angle theta (sphere tracing on the signed distances), far box
        faces colored by wall type. The clipping plane also cuts the fixed obstacles (open a closed casing), not
        the rotor.

        **Outputs**

        - img[:h, :w] and zbuf[:h, :w] written (zbuf = hit distance, 1e30 outside the box)
        """
        n = ti.static(sdf.shape)
        light = ti.Vector([0.4, 0.8, 0.45]).normalized()
        for row, col in ti.ndrange(h, w):
            d = (fwd + (2.0 * (col + 0.5) / w - 1.0) * th * (w / h) * right
                 + (1.0 - 2.0 * (row + 0.5) / h) * th * up).normalized()
            inv = 1.0 / ti.select(ti.abs(d) < 1e-12, 1e-12, d)
            t0, t1 = (0.0 - eye) * inv, (ext - eye) * inv
            tn = ti.max(ti.min(t0, t1).max(), 0.0)
            tf = ti.max(t0, t1).min()
            c = ti.Vector([0.06, 0.06, 0.08]) + 0.05 * (1.0 - (row + 0.5) / h)
            z = 1e30
            self.odepth[row, col] = 1e30
            if tn < tf:
                hit = False
                skip = False                                   # translucide : obstacle fixe déjà rencontré
                if use_sdf == 1 or has_rot == 1:              # obstacles et rotor : sphere tracing sur les distances
                    t = tn
                    for _ in range(160):
                        if not hit and t < tf:
                            p = eye + t * d
                            phi, grad = 1e3, ti.Vector([0.0, 1.0, 0.0])
                            if use_sdf == 1 and not skip:
                                phi, grad = sdf_at(sdf, p, 1.0 / dx)
                                for a in ti.static(range(3)):            # coupe : obstacle ∩ demi-espace gardé
                                    if ca == a:
                                        hc = p[a] - cpos if ckeep == 1 else cpos - p[a]
                                        if hc > phi:
                                            phi = hc
                                            grad = ti.Vector.unit(3, a, ti.f32) * (1.0 if ckeep == 1 else -1.0)
                            rot = False
                            if has_rot == 1:
                                pr, gr = rotor_sdf(rsdf, p, rc[None], axis, theta, 1.0 / dx)
                                if pr < phi:
                                    phi, grad, rot = pr, gr, True
                            if phi < 0.3 * dx:
                                nrm = grad.normalized()
                                lam = ti.max(nrm.dot(light), 0.0)
                                base = ti.Vector([0.80, 0.55, 0.25]) if rot else ti.Vector([0.45, 0.45, 0.47])
                                if translucent == 1 and not rot:     # mémorisé, on continue derrière
                                    self.odepth[row, col] = t
                                    self.ocol[row, col] = base * (0.35 + 0.65 * lam)
                                    skip = True
                                    phi = 0.4 * dx
                                else:
                                    hit = True
                                    c = base * (0.35 + 0.65 * lam)
                                    z = t
                            t += ti.max(phi, 0.4 * dx)
                if not hit:                                   # face du fond : type de paroi
                    p = eye + tf * d
                    best, side = 1e30, 0
                    for a in ti.static(range(3)):
                        if ti.abs(p[a]) < best:
                            best, side = ti.abs(p[a]), 2 * a
                        if ti.abs(p[a] - ext[a]) < best:
                            best, side = ti.abs(p[a] - ext[a]), 2 * a + 1
                    cell = ti.math.clamp((p / dx).cast(int), 0, ti.Vector(n) - 1)
                    t_wall = 0
                    shade = 1.0
                    for s in ti.static(range(6)):
                        a, plus, ts = ti.static(FRAMES[3][s])
                        if side == s:
                            ka, kb = cell[ts[0]], cell[ts[1]]
                            ok = bound <= ka < n[ts[0]] - bound and bound <= kb < n[ts[1]] - bound
                            if ok:
                                t_wall = wall_type[s, ka, kb]
                            shade = ti.static((0.85, 0.85, 1.0, 0.75, 0.92, 0.92)[s])
                    c = _wall_color(t_wall) * shade
                    gx = ti.abs(p / dx - ti.round(p / dx))            # quadrillage discret (cellules de 1 dx)
                    edge = 0
                    for a in ti.static(range(3)):
                        if a != side // 2 and gx[a] < 0.04:
                            edge = 1
                    if edge == 1:
                        c *= 0.85
                    z = tf
            img[row, col] = ti.cast(ti.math.clamp(c, 0.0, 1.0) * 255, ti.u8)
            self.zbuf[row, col] = z

    # ------------------------------------------------------------ 2. particules en sphères
    @ti.func
    def _proj(self, xp, eye, fwd, right, up, th, w, h):
        """Pixel position, depth and focal (px / m at unit depth) of a point."""
        v = xp - eye
        z = v.dot(fwd)
        zs = ti.max(z, 1e-6)
        sx = v.dot(right) / (zs * th * (w / h))
        sy = v.dot(up) / (zs * th)
        return (sx + 1.0) * 0.5 * w, (1.0 - sy) * 0.5 * h, z, 0.5 * h / th

    @ti.func
    def _clipped(self, xp, ca, cpos, ckeep) -> bool:
        """Whether a particle is hidden by the clipping plane."""
        hid = False
        for a in ti.static(range(3)):
            if ca == a:
                hid = xp[a] > cpos if ckeep == 1 else xp[a] < cpos
        return hid

    @ti.kernel
    def _depth(self, x: ti.template(), flag: ti.template(), want: int, w: int, h: int,
               eye: ti.types.vector(3, ti.f32), fwd: ti.types.vector(3, ti.f32), right: ti.types.vector(3, ti.f32),
               up: ti.types.vector(3, ti.f32), th: float, radius: float, ca: int, cpos: float, ckeep: int):
        """Depth pass: nearest sphere surface per pixel (atomic_min on zbuf).

        **Inputs**

        - `x` : vec3 field positions ; `flag` : i32 field ; `want` : int (fluid: alive == 1 ; solid: any, want 0
          means every particle is drawn)
        """
        for p in x:
            if (want == 0 or flag[p] == want) and not self._clipped(x[p], ca, cpos, ckeep):
                cx, cy, z, foc = self._proj(x[p], eye, fwd, right, up, th, w, h)
                if z > radius:
                    rp = radius * foc / z
                    r_i = ti.min(int(rp) + 1, 24)
                    for a, b in ti.ndrange((-r_i, r_i + 1), (-r_i, r_i + 1)):
                        col, row = int(cx) + a, int(cy) + b
                        if 0 <= col < w and 0 <= row < h:
                            q2 = ((col + 0.5 - cx) ** 2 + (row + 0.5 - cy) ** 2) / (rp * rp)
                            if q2 < 1.0:
                                ti.atomic_min(self.zbuf[row, col], z - radius * ti.sqrt(1.0 - q2))

    @ti.func
    def _splat(self, img: ti.template(), xp, color, w, h, eye, fwd, right, up, th, radius):
        """Shade the pixels where this sphere is the nearest surface."""
        light = ti.Vector([0.45, 0.75, 0.5]).normalized()
        cx, cy, z, foc = self._proj(xp, eye, fwd, right, up, th, w, h)
        if z > radius:
            rp = radius * foc / z
            r_i = ti.min(int(rp) + 1, 24)
            for a, b in ti.ndrange((-r_i, r_i + 1), (-r_i, r_i + 1)):
                col, row = int(cx) + a, int(cy) + b
                if 0 <= col < w and 0 <= row < h:
                    ux, uy = (col + 0.5 - cx) / rp, (row + 0.5 - cy) / rp
                    q2 = ux * ux + uy * uy
                    if q2 < 1.0:
                        zs = z - radius * ti.sqrt(1.0 - q2)
                        if zs <= self.zbuf[row, col] + 1e-6 * z:
                            nz = ti.sqrt(1.0 - q2)
                            nrm = ux * right - uy * up - nz * fwd              # normale dans le repère monde
                            lam = ti.max(nrm.dot(light), 0.0)
                            c = color * (0.35 + 0.65 * lam) + 0.25 * lam ** 24   # reflet
                            img[row, col] = ti.cast(ti.math.clamp(c, 0.0, 1.0) * 255, ti.u8)

    @ti.kernel
    def _shade_fluid(self, img: ti.template(), x: ti.template(), alive: ti.template(), sc: ti.template(),
                     mode: int, signed: int, inv_smax: float, base: ti.types.vector(3, ti.f32), w: int, h: int,
                     eye: ti.types.vector(3, ti.f32), fwd: ti.types.vector(3, ti.f32),
                     right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32), th: float, radius: float,
                     ca: int, cpos: float, ckeep: int):
        """Color pass of the fluid spheres (palette of the 2D view)."""
        for p in x:
            if alive[p] == 1 and not self._clipped(x[p], ca, cpos, ckeep):
                color = particle_color(sc[p], mode, signed, inv_smax, base)
                self._splat(img, x[p], color, w, h, eye, fwd, right, up, th, radius)

    @ti.kernel
    def _shade_solid(self, img: ti.template(), x: ti.template(), col_s: ti.template(), w: int, h: int,
                     eye: ti.types.vector(3, ti.f32), fwd: ti.types.vector(3, ti.f32),
                     right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32), th: float, radius: float,
                     ca: int, cpos: float, ckeep: int):
        """Color pass of the solid spheres (damage / stretch colors)."""
        for p in x:
            if not self._clipped(x[p], ca, cpos, ckeep):
                self._splat(img, x[p], col_s[p], w, h, eye, fwd, right, up, th, radius)
