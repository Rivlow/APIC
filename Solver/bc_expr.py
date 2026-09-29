# Solver/bc_expr.py -- Conditions aux limites données par une expression de (x, y, z, t) et de constantes.
#
# Une valeur imposée (vitesse d'entrée vx / vy / vz, pression de sortie) est un nombre ou une chaîne, par exemple
# "rho*g*(H - y)" ou "U*(1 - exp(-t/T))". Variables : x, y, z (unités domaine, centre de la cellule de mur sur le
# mur ; z = 0 en 2D), t (temps simulé, s). Constantes : rho, g, pi, Lx, Ly, Lz, dx et celles de
# SimulationRunner.consts. Fonctions : min,
# max, abs, sqrt, exp, log, sin, cos, tan, tanh ; comparaisons < <= > >= valant 1 ou 0 (ex. "U*(y < H)").
# Rien d'autre n'est accepté (analyse de l'AST, pas d'eval libre).
#
# Sans t : évaluée une fois (numpy) à la rastérisation des parois, coût nul pendant le calcul.
# Avec t : compilée en un kernel Taichi (module généré dans un fichier, Taichi lit le source des kernels) qui
# réécrit wall_v / wall_p sur le GPU à chaque sous-pas, sans aucune copie CPU <-> GPU.

import ast
import hashlib
import importlib.util
import os
import tempfile

import numpy as np

VARS = ("x", "y", "z", "t")
FUNCS = {"min": ("np.minimum", "ti.min"), "max": ("np.maximum", "ti.max"), "abs": ("np.abs", "ti.abs"),
         "sqrt": ("np.sqrt", "ti.sqrt"), "exp": ("np.exp", "ti.exp"), "log": ("np.log", "ti.log"),
         "sin": ("np.sin", "ti.sin"), "cos": ("np.cos", "ti.cos"), "tan": ("np.tan", "ti.tan"),
         "tanh": ("np.tanh", "ti.tanh")}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)          # comparaison : 1 si vraie, 0 sinon (profils par morceaux)


def is_expr(value) -> bool:
    """Whether a boundary value is an expression (str) rather than a number.

    **Inputs**

    - `value` : float | int | str

    **Outputs**

    - bool
    """
    return isinstance(value, str)


def check(src: str) -> None:
    """Validate an expression's syntax and allowed operations (names are checked later, with the constants).

    **Inputs**

    - `src` : str expression

    **Outputs**

    - None ; raises ValueError if the expression is not allowed
    """
    _translate(src, None, 0)


def uses_time(src: str) -> bool:
    """Whether an expression depends on t (then it is re-evaluated on the GPU at each substep).

    **Inputs**

    - `src` : str expression

    **Outputs**

    - bool
    """
    tree = ast.parse(src.strip(), mode="eval")
    return any(isinstance(n, ast.Name) and n.id == "t" for n in ast.walk(tree))


def _translate(src: str, consts: dict | None, target: int) -> str:
    """Translate an expression to numpy (target 0) or Taichi (target 1) source, constants inlined.

    **Inputs**

    - `src` : str expression
    - `consts` : dict[str, float] | None (None : only syntax is checked, names are not resolved)
    - `target` : int 0 = numpy, 1 = Taichi

    **Outputs**

    - str translated expression
    """
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression invalide {src!r} : {exc.msg}") from None

    def tr(n) -> str:
        if isinstance(n, ast.Expression):
            return tr(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return repr(float(n.value))
        if isinstance(n, ast.Name):
            if n.id in VARS:
                return n.id
            if consts is None:
                return "0.0"
            if n.id not in consts:
                raise ValueError(f"nom inconnu {n.id!r} dans {src!r} (variables {VARS}, constantes "
                                 f"{sorted(consts)})")
            return repr(float(consts[n.id]))
        if isinstance(n, ast.BinOp) and isinstance(n.op, _BINOPS):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**", ast.Mod: "%"}[type(n.op)]
            return f"({tr(n.left)} {op} {tr(n.right)})"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return f"({'-' if isinstance(n.op, ast.USub) else '+'}{tr(n.operand)})"
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], _CMPOPS):
            op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(n.ops[0])]
            cmp = f"({tr(n.left)} {op} {tr(n.comparators[0])})"
            return f"({cmp} * 1.0)" if target == 0 else f"ti.cast({cmp}, ti.f32)"   # vrai = 1, faux = 0
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
            want = 2 if n.func.id in ("min", "max") else 1
            if len(n.args) != want:
                raise ValueError(f"{n.func.id} attend {want} argument(s) dans {src!r}")
            return f"{FUNCS[n.func.id][target]}({', '.join(tr(a) for a in n.args)})"
        raise ValueError(f"opération non autorisée dans {src!r} : {ast.unparse(n) if hasattr(ast, 'unparse') else n}")

    return tr(tree)


def evaluate(src: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float, consts: dict) -> np.ndarray:
    """Evaluate an expression with numpy at wall-cell positions.

    **Inputs**

    - `src` : str expression
    - `x`, `y`, `z` : np.ndarray f64 (m,) positions (domain units ; z = 0 in 2D)
    - `t` : float time
    - `consts` : dict[str, float]

    **Outputs**

    - np.ndarray f64 (m,)
    """
    code = _translate(src, consts, 0)
    val = eval(code, {"np": np, "__builtins__": {}}, {"x": x, "y": y, "z": z, "t": t})   # code issu d'un AST filtré
    return np.broadcast_to(np.asarray(val, dtype=np.float64), np.shape(x)).copy()


def build_kernel(sources: list[str], consts: dict, dim: int = 2):
    """Generate and import a Taichi kernel that re-evaluates time-dependent wall values.

    **Inputs**

    - `sources` : list[str] expressions (index = expression id stored in the wall table)
    - `consts` : dict[str, float]
    - `dim` : int 2 or 3

    **Outputs**

    - Taichi kernel `eval_bc(wall_e, wall_v, wall_p, wall_d, t, dx, nx, ny, nz)` (see generated docstring)

    **Note** : Taichi reads kernel source with inspect, so the module is written to a cached file and imported.
    """
    exprs = [_translate(s, consts, 1) for s in sources]
    branches = "\n".join(f"    {'if' if i == 0 else 'elif'} e == {i}:\n        r = {c}" for i, c in enumerate(exprs))
    code = f'''import taichi as ti

DIM = {dim}


@ti.func
def _expr(e: int, x: float, y: float, z: float, t: float) -> float:
    r = 0.0
{branches}
    return r


@ti.kernel
def eval_bc(wall_e: ti.template(), wall_v: ti.template(), wall_p: ti.template(), wall_d: ti.template(),
            t: float, dx: float, nx: int, ny: int, nz: int):
    """Re-evaluate time-dependent wall values (v..., p) at wall-cell positions (side, ka, kb)."""
    n = ti.Vector([nx, ny, nz])
    for s, ka, kb in wall_d:
        a = s // 2
        ta, tb = 1, 2                                 # axes tangents (ordre croissant)
        if a == 1:
            ta = 0
        elif a == 2:
            ta, tb = 0, 1
        if ti.static(DIM == 2):
            ta = 1 - a
        la, lb = 1, 1
        for c in ti.static(range(3)):
            if c == ta:
                la = n[c]
            if c == tb and ti.static(DIM == 3):
                lb = n[c]
        if ka < la and kb < lb:
            d = wall_d[s, ka, kb]
            p = ti.Vector([0.0, 0.0, 0.0])
            for c in ti.static(range(3)):
                if c == a:
                    p[c] = d * dx if s % 2 == 0 else (n[c] - d) * dx
                elif c == ta:
                    p[c] = (ka + 0.5) * dx
                elif c == tb and ti.static(DIM == 3):
                    p[c] = (kb + 0.5) * dx
            ids = wall_e[s, ka, kb]
            for ch in ti.static(range(DIM)):
                if ids[ch] >= 0:
                    wall_v[s, ka, kb][ch] = _expr(ids[ch], p[0], p[1], p[2], t)
            if ids[DIM] >= 0:
                wall_p[s, ka, kb] = _expr(ids[DIM], p[0], p[1], p[2], t)
'''
    tag = hashlib.sha1(code.encode()).hexdigest()[:16]
    folder = os.path.join(tempfile.gettempdir(), "apic_bc_kernels")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"bc_{tag}.py")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    spec = importlib.util.spec_from_file_location(f"apic_bc_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.eval_bc
