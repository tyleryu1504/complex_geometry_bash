"""
Symbolic geometry engine for complex-number bashing in the circumcenter model.

The implementation follows the behavior spec outlined in AGENTS.md and the
user-provided requirements.  Points are represented by independent symbols
(z_X, zb_X).  Constraints are maintained as polynomials, and the engine learns
conjugate substitution rules whenever a single-conjugate equation is detected.
""" 
"""
test
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import sympy as sp


class GeometryError(RuntimeError):
    """Raised when a construction or solve step fails."""


@dataclass
class PointRecord:
    z: sp.Symbol
    zb: sp.Symbol


@dataclass
class UnitTriangleConfig:
    points: Dict[str, str]
    roots: Dict[str, str]


@dataclass(frozen=True)
class Line:
    """
    Algebraic representation of a line alpha*z + beta*zb + gamma = 0.

    The coefficients may depend on previously registered point symbols; the context tuple
    records any labels used to derive the line for traceability.
    """

    alpha: sp.Expr
    beta: sp.Expr
    gamma: sp.Expr
    context: Tuple[str, ...] = ()

    def evaluate(self, z_expr: sp.Expr, zb_expr: sp.Expr) -> sp.Expr:
        """Return the symbolic value of the line equation at (z_expr, zb_expr)."""
        return sp.simplify(self.alpha * z_expr + self.beta * zb_expr + self.gamma)


@dataclass(frozen=True)
class Circle:
    """
    Representation of a circle determined by its center and squared radius.
    """

    center: str
    radius_squared: sp.Expr
    context: Tuple[str, ...] = ()

    def evaluate(
        self, engine: "GeometryEngine", z_expr: sp.Expr, zb_expr: sp.Expr
    ) -> sp.Expr:
        """Return the circle equation evaluated at (z_expr, zb_expr)."""
        z_center = engine.z(self.center)
        zb_center = engine.zb(self.center)
        return sp.simplify(
            (z_expr - z_center) * (zb_expr - zb_center) - self.radius_squared
        )


class GeometryEngine:
    """
    Geometry Engine working in the complex plane with symbolic coordinates.

    Each point has two independent symbols (z_X, zb_X).  The engine stores
    constraints as polynomials and maintains a substitution system composed of:

    * Value substitutions (e.g. origin at 0)
    * Assignments produced by constructions (rational expressions)
    * Explicit model substitutions (for example zb_U -> 1/z_U on the unit circle)
    * Assignments produced by explicit point calculations
    """

    def __init__(self) -> None:
        self.points: Dict[str, PointRecord] = {}
        self.constraints: List[sp.Expr] = []
        self.value_subs: Dict[sp.Symbol, sp.Expr] = {}
        self.point_assignments: Dict[sp.Symbol, sp.Expr] = {}
        self.model_subs: Dict[sp.Symbol, sp.Expr] = {}
        # Kept for CLI/test compatibility. Constraints no longer populate this map.
        self.learned_subs: Dict[sp.Symbol, sp.Expr] = {}
        self.unit_circle_points: Set[str] = set()
        self.lines: Dict[str, Line] = {}
        self.line_parameters: Dict[str, Tuple[sp.Symbol, sp.Symbol, sp.Symbol]] = {}
        self.circles: Dict[str, Circle] = {}
        self.distinct_pairs: Set[FrozenSet[str]] = set()
        self._main_unit_triangle: Optional[UnitTriangleConfig] = None
        self._latex_symbol_names: Dict[sp.Symbol, str] = {}
        self._text_symbol_replacements: Dict[sp.Symbol, sp.Expr] = {}

        # Declare origin O with fixed coordinates at 0
        self.add_point("O")
        self._register_origin()

    # ------------------------------------------------------------------
    # Point management helpers
    # ------------------------------------------------------------------
    def add_point(self, name: str) -> None:
        """
        Register a point with independent (z, zb) symbols.

        Parameters
        ----------
        name:
            Label of the point (must be unique).
        """
        if name in self.points:
            return

        z_symbol = sp.Symbol(f"z_{name}")
        zb_symbol = sp.Symbol(f"zb_{name}")
        self.points[name] = PointRecord(z=z_symbol, zb=zb_symbol)
        self._register_display_symbols(name, z_symbol, zb_symbol)

    def _register_display_symbols(
        self, name: str, z_symbol: sp.Symbol, zb_symbol: sp.Symbol
    ) -> None:
        """Cache pretty-print mappings for the point's coordinate symbols."""
        base_symbol = sp.Symbol(name)
        latex_label = sp.latex(base_symbol)
        self._latex_symbol_names[z_symbol] = latex_label
        self._latex_symbol_names[zb_symbol] = rf"\overline{{{latex_label}}}"
        self._text_symbol_replacements[z_symbol] = base_symbol
        self._text_symbol_replacements[zb_symbol] = sp.conjugate(base_symbol)

    def ensure_point(self, name: str) -> None:
        """Ensure a point is registered; raise an informative error otherwise."""
        if name not in self.points:
            raise GeometryError(f"Point '{name}' is not registered.")

    def _register_origin(self) -> None:
        """Apply the origin substitution rules (O fixed at 0)."""
        self.ensure_point("O")
        origin = self.points["O"]
        self.value_subs[origin.z] = sp.Integer(0)
        self.value_subs[origin.zb] = sp.Integer(0)

    def add_unit_circle(self, name: str) -> None:
        """Declare that point name lies on the unit circle."""
        self.ensure_point(name)
        if name in self.unit_circle_points:
            return
        record = self.points[name]
        constraint = record.z * record.zb - 1
        self.model_subs[record.zb] = sp.simplify(1 / record.z)
        self.add_constraint(constraint)
        self.unit_circle_points.add(name)

    # ------------------------------------------------------------------
    # Symbol and expression helpers
    # ------------------------------------------------------------------
    def z_symbol(self, name: str) -> sp.Symbol:
        self.ensure_point(name)
        return self.points[name].z

    def zb_symbol(self, name: str) -> sp.Symbol:
        self.ensure_point(name)
        return self.points[name].zb

    def z(self, name: str) -> sp.Expr:
        """Return the (possibly substituted) expression for z_name."""
        symbol = self.z_symbol(name)
        expr = self.point_assignments.get(symbol, symbol)
        return self._apply_all(expr)

    def zb(self, name: str) -> sp.Expr:
        """Return the (possibly substituted) expression for zb_name."""
        symbol = self.zb_symbol(name)
        expr = self.point_assignments.get(symbol, symbol)
        return self._apply_all(expr)

    def _substitution_map(self) -> Dict[sp.Symbol, sp.Expr]:
        """Compose the global substitution map in precedence order."""
        subs: Dict[sp.Symbol, sp.Expr] = {}
        subs.update(self.value_subs)
        subs.update(self.point_assignments)
        subs.update(self.model_subs)
        return subs

    def _apply_all(self, expr: sp.Expr) -> sp.Expr:
        """
        Apply all known substitutions (value, assignments, and model rules).

        Multiple passes are used to ensure transitive substitutions settle.
        """
        if expr is None:
            return expr

        current = sp.simplify(expr)
        if current == 0:
            return sp.Integer(0)

        subs = self._substitution_map()
        for _ in range(4):
            updated = current.subs(subs, simultaneous=False)
            updated = sp.simplify(updated)
            if updated == current:
                break
            current = updated
        return sp.simplify(current)

    # ------------------------------------------------------------------
    # Constraint handling
    # ------------------------------------------------------------------
    def add_constraint(self, constraint: sp.Expr) -> None:
        """
        Record a polynomial constraint (assumed == 0).
        """

        raw_constraint = sp.expand(constraint)
        self.constraints.append(raw_constraint)

    # ------------------------------------------------------------------
    # Predicate polynomials
    # ------------------------------------------------------------------
    def collinear_poly(self, A: str, B: str, C: str, *, raw: bool = False) -> sp.Expr:
        """
        Return the polynomial witnessing collinearity of points A, B, C.
        """
        zA, zB, zC = self.z_symbol(A), self.z_symbol(B), self.z_symbol(C)
        zbA, zbB, zbC = self.zb_symbol(A), self.zb_symbol(B), self.zb_symbol(C)
        poly = (zA - zC) * (zbB - zbC) - (zbA - zbC) * (zB - zC)
        return poly if raw else sp.expand(poly)

    def perpendicular_poly(
        self, A: str, B: str, C: str, D: str, *, raw: bool = False
    ) -> sp.Expr:
        """
        Return the polynomial encoding AB ⟂ CD.
        """
        zA, zB = self.z_symbol(A), self.z_symbol(B)
        zC, zD = self.z_symbol(C), self.z_symbol(D)
        zbA, zbB = self.zb_symbol(A), self.zb_symbol(B)
        zbC, zbD = self.zb_symbol(C), self.zb_symbol(D)
        poly = (zA - zB) * (zbC - zbD) + (zbA - zbB) * (zC - zD)
        return poly if raw else sp.expand(poly)

    def concyclic_poly(
        self, A: str, B: str, C: str, D: str, *, raw: bool = False
    ) -> sp.Expr:
        """
        Return a polynomial enforcing that A, B, C, D lie on a common circle.

        The polynomial is obtained by eliminating the circle center U from the
        system |A-U|^2 = |B-U|^2, |A-U|^2 = |C-U|^2, |A-U|^2 = |D-U|^2. The first
        two equations are solved for U; the solution is substituted into the
        third, and denominators are cleared to yield a single polynomial in the
        point coordinates.
        """
        zA, zB, zC, zD = (
            self.z_symbol(A),
            self.z_symbol(B),
            self.z_symbol(C),
            self.z_symbol(D),
        )
        zbA, zbB, zbC, zbD = (
            self.zb_symbol(A),
            self.zb_symbol(B),
            self.zb_symbol(C),
            self.zb_symbol(D),
        )

        zU = sp.Symbol(f"z_circ_{A}{B}{C}", complex=True)
        zbU = sp.Symbol(f"zb_circ_{A}{B}{C}", complex=True)

        eq1 = sp.expand((zA - zU) * (zbA - zbU) - (zB - zU) * (zbB - zbU))
        eq2 = sp.expand((zA - zU) * (zbA - zbU) - (zC - zU) * (zbC - zbU))

        solutions = sp.solve((sp.Eq(eq1, 0), sp.Eq(eq2, 0)), (zU, zbU), dict=True)
        if not solutions:
            raise GeometryError("Failed to derive circumcenter for concyclicity test.")

        solution = solutions[0]
        zU_expr = sp.simplify(solution[zU])
        zbU_expr = sp.simplify(solution[zbU])

        eq3 = sp.simplify(
            (zA - zU_expr) * (zbA - zbU_expr) - (zD - zU_expr) * (zbD - zbU_expr)
        )
        numerator, denominator = sp.fraction(sp.simplify(eq3))
        poly = sp.expand(numerator)
        return poly if raw else sp.expand(poly)

    def angle_value_poly(
        self, A: str, B: str, C: str, angle_radians: sp.Expr, *, raw: bool = False
    ) -> sp.Expr:
        """
        Return polynomial enforcing that angle ABC equals the provided angle (in radians).

        Uses the identity that (z_A - z_B)/(z_C - z_B) / exp(i*theta) is real.
        """
        zA, zB, zC = self.z_symbol(A), self.z_symbol(B), self.z_symbol(C)
        zbA, zbB, zbC = self.zb_symbol(A), self.zb_symbol(B), self.zb_symbol(C)
        w = sp.exp(sp.I * angle_radians)
        w_conj = sp.exp(-sp.I * angle_radians)
        poly = w_conj * (zA - zB) * (zbC - zbB) - w * (zbA - zbB) * (zC - zB)
        return poly if raw else sp.expand(poly)

    # ------------------------------------------------------------------
    # Rational invariants
    # ------------------------------------------------------------------
    def cross_ratio(
        self, A: str, B: str, C: str, D: str, *, apply_subs: bool = True
    ) -> sp.Expr:
        """
        Return the cross ratio (A, B; C, D) given by ((z_A - z_C)(z_B - z_D)) / ((z_A - z_D)(z_B - z_C)).

        Parameters
        ----------
        apply_subs:
            When True, apply any known substitutions (assignments, learned rules) to the result.
        """
        zA, zB, zC, zD = (
            self.z_symbol(A),
            self.z_symbol(B),
            self.z_symbol(C),
            self.z_symbol(D),
        )
        expr = sp.simplify((zA - zC) * (zB - zD) / ((zA - zD) * (zB - zC)))
        return self._apply_all(expr) if apply_subs else expr

    def angle(self, A: str, B: str, C: str, *, apply_subs: bool = True):
        """
        Return the angle ABC given by ((z_A - z_B)/conj(z_A - z_B)) / ((z_C - z_B)/conj(z_B - z_C)).

        Parameters
        ----------
        apply_subs:
            When True, apply any known substitutions (assignments, learned rules) to the result.
        """
        zA, zB, zC = (
            self.z_symbol(A),
            self.z_symbol(B),
            self.z_symbol(C),
        )
        zbA, zbB, zbC = (
            self.zb_symbol(A),
            self.zb_symbol(B),
            self.zb_symbol(C),
        )
        expr = ((zA - zB) / (zbA - zbB)) / ((zC - zB) / (zbC - zbB))
        return self._apply_all(expr) if apply_subs else expr

    def angle_bisector_either_poly(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        *,
        raw: bool = False,
    ) -> sp.Expr:
        """
        Return a polynomial ensuring D lies on either bisector of angle BAC.

        Derived from the identity (z_D - z_A)^2 / conj((z_D - z_A)^2) =
        (z_B - z_A)(z_C - z_A) / conj((z_B - z_A)(z_C - z_A)).
        """
        zA, zB, zC, zD = (
            self.z_symbol(A),
            self.z_symbol(B),
            self.z_symbol(C),
            self.z_symbol(D),
        )
        zbA, zbB, zbC, zbD = (
            self.zb_symbol(A),
            self.zb_symbol(B),
            self.zb_symbol(C),
            self.zb_symbol(D),
        )
        lhs = (zD - zA) ** 2 * (zbB - zbA) * (zbC - zbA)
        rhs = (zbD - zbA) ** 2 * (zB - zA) * (zC - zA)
        poly = sp.simplify(lhs - rhs)
        return poly if raw else sp.expand(poly)

    def add_angle_bisector_either(self, A: str, B: str, C: str, D: str) -> None:
        """Store the constraint that D lies on either angle bisector of BAC."""
        poly = self.angle_bisector_either_poly(A, B, C, D)
        self.add_constraint(poly)

    def triangle_similarity_polys(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        F: str,
        *,
        directed: bool = True,
        raw: bool = False,
    ) -> Tuple[sp.Expr, ...]:
        """
        Return polynomial constraints enforcing similarity between triangles ABC and DEF.

        Parameters
        ----------
        directed:
            When True, enforce orientation-preserving similarity.  When False, allow mirrored
            similarity (orientation-reversing).
        """
        zA, zB, zC = self.z_symbol(A), self.z_symbol(B), self.z_symbol(C)
        zD, zE, zF = self.z_symbol(D), self.z_symbol(E), self.z_symbol(F)
        zbA, zbB, zbC = self.zb_symbol(A), self.zb_symbol(B), self.zb_symbol(C)
        zbD, zbE, zbF = self.zb_symbol(D), self.zb_symbol(E), self.zb_symbol(F)

        if directed:
            eq_primary = (zA - zB) * (zE - zF) - (zB - zC) * (zD - zE)
            eq_conjugate = (zbA - zbB) * (zbE - zbF) - (zbB - zbC) * (zbD - zbE)
        else:
            eq_primary = (zA - zB) * (zbE - zbF) - (zB - zC) * (zbD - zbE)
            eq_conjugate = (zbA - zbB) * (zE - zF) - (zbB - zbC) * (zD - zE)

        if not raw:
            eq_primary = sp.expand(eq_primary)
            eq_conjugate = sp.expand(eq_conjugate)

        return eq_primary, eq_conjugate

    def triangle_congruence_polys(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        F: str,
        *,
        directed: bool = True,
        raw: bool = False,
    ) -> Tuple[sp.Expr, ...]:
        """
        Return polynomial constraints enforcing triangle ABC congruent to triangle DEF.

        Congruence is modeled as similarity plus equality of a reference side length.  When
        ``directed`` is False, orientation reversal (mirror) is permitted.
        """
        similarity_eqs = self.triangle_similarity_polys(
            A,
            B,
            C,
            D,
            E,
            F,
            directed=directed,
            raw=True,
        )

        zA, zB = self.z_symbol(A), self.z_symbol(B)
        zD, zE = self.z_symbol(D), self.z_symbol(E)
        zbA, zbB = self.zb_symbol(A), self.zb_symbol(B)
        zbD, zbE = self.zb_symbol(D), self.zb_symbol(E)

        length_eq = (zA - zB) * (zbA - zbB) - (zD - zE) * (zbD - zbE)

        if not raw:
            similarity_eqs = tuple(sp.expand(eq) for eq in similarity_eqs)
            length_eq = sp.expand(length_eq)

        return (*similarity_eqs, length_eq)

    def circumcenter_polys(
        self, A: str, B: str, C: str, U: str, *, raw: bool = False
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Return the two polynomials that enforce U as the circumcenter of triangle ABC.

        |A-U|^2 = |B-U|^2 and |A-U|^2 = |C-U|^2, expressed as polynomial equalities.
        """
        zU = self.z_symbol(U)
        zbU = self.zb_symbol(U)

        def squared_distance(zX: sp.Expr, zbX: sp.Expr) -> sp.Expr:
            return (zX - zU) * (zbX - zbU)

        zA, zB, zC = self.z_symbol(A), self.z_symbol(B), self.z_symbol(C)
        zbA, zbB, zbC = self.zb_symbol(A), self.zb_symbol(B), self.zb_symbol(C)

        dist_A = squared_distance(zA, zbA)
        dist_B = squared_distance(zB, zbB)
        dist_C = squared_distance(zC, zbC)

        eq1 = dist_A - dist_B
        eq2 = dist_A - dist_C
        if not raw:
            eq1 = sp.expand(eq1)
            eq2 = sp.expand(eq2)
        return eq1, eq2

    def midpoint_polys(
        self, P: str, Q: str, M: str, *, raw: bool = False
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Polynomials enforcing that M is the midpoint of segment PQ.
        """
        zP, zQ, zM = self.z_symbol(P), self.z_symbol(Q), self.z_symbol(M)
        zbP, zbQ, zbM = self.zb_symbol(P), self.zb_symbol(Q), self.zb_symbol(M)
        eq_z = 2 * zM - zP - zQ
        eq_zb = 2 * zbM - zbP - zbQ
        if not raw:
            eq_z = sp.expand(eq_z)
            eq_zb = sp.expand(eq_zb)
        return eq_z, eq_zb

    def centroid_polys(
        self, A: str, B: str, C: str, G: str, *, raw: bool = False
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Polynomials enforcing that G is the centroid of triangle ABC.
        """
        for label in (A, B, C, G):
            self.ensure_point(label)

        zA, zB, zC, zG = (
            self.z_symbol(A),
            self.z_symbol(B),
            self.z_symbol(C),
            self.z_symbol(G),
        )
        zbA, zbB, zbC, zbG = (
            self.zb_symbol(A),
            self.zb_symbol(B),
            self.zb_symbol(C),
            self.zb_symbol(G),
        )

        eq_z = zA + zB + zC - 3 * zG
        eq_zb = zbA + zbB + zbC - 3 * zbG

        if not raw:
            eq_z = sp.expand(eq_z)
            eq_zb = sp.expand(eq_zb)
        return eq_z, eq_zb

    def point_reflection_polys(
        self, P: str, O: str, Q: str, *, raw: bool = False
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Polynomials enforcing that Q is the reflection of P across point O.
        """
        zP, zO, zQ = self.z_symbol(P), self.z_symbol(O), self.z_symbol(Q)
        zbP, zbO, zbQ = self.zb_symbol(P), self.zb_symbol(O), self.zb_symbol(Q)
        eq_z = zQ + zP - 2 * zO
        eq_zb = zbQ + zbP - 2 * zbO
        if not raw:
            eq_z = sp.expand(eq_z)
            eq_zb = sp.expand(eq_zb)
        return eq_z, eq_zb

    def line_reflection_polys(
        self, P: str, A: str, B: str, Q: str, *, raw: bool = False
    ) -> Tuple[sp.Expr, ...]:
        """
        Polynomials enforcing that Q is the reflection of P across line AB.
        """
        eq_perp = self.perpendicular_poly(A, B, P, Q, raw=True)

        midpoint_label = self._midpoint_label(P, Q)
        self.add_point(midpoint_label)
        eq_mid_z, eq_mid_zb = self.midpoint_polys(P, Q, midpoint_label, raw=True)
        collinear_midpoint = self.collinear_poly(midpoint_label, A, B, raw=True)

        if not raw:
            eq_perp = sp.expand(eq_perp)
            eq_mid_z = sp.expand(eq_mid_z)
            eq_mid_zb = sp.expand(eq_mid_zb)
            collinear_midpoint = sp.expand(collinear_midpoint)

        return eq_perp, eq_mid_z, eq_mid_zb, collinear_midpoint

    def projection_to_line_polys(
        self,
        P: str,
        A: str,
        B: str,
        H: str,
        *,
        raw: bool = False,
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Polynomials enforcing that H is the orthogonal projection of P onto line AB.

        The projection is modeled as the intersection of line AB with the perpendicular through P.
        """
        if A == B:
            raise GeometryError(
                "Projection requires two distinct points to define line AB."
            )

        eq_perp = self.perpendicular_poly(A, B, P, H, raw=True)
        eq_collinear = self.collinear_poly(H, A, B, raw=True)

        if not raw:
            eq_perp = sp.expand(eq_perp)
            eq_collinear = sp.expand(eq_collinear)

        return eq_perp, eq_collinear

    def equal_angle_poly(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        F: str,
        *,
        apply_subs: bool = False,
    ) -> sp.Expr:
        """
        Polynomial enforcing that ABC=DEF( directed angle ).

        The condition is expressed through complex directions:

            (z_A - z_B)(z_E - z_A) / conj((z_D - z_A)(z_E - z_A))
            = (z_B - z_A)(z_C - z_A) / conj((z_B - z_A)(z_C - z_A))

        which simplifies to the cross-multiplication below.
        """
        expr = self.angle(A, B, C) - self.angle(D, E, F)
        return self._apply_all(expr) if apply_subs else expr

    def equal_length_poly(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        *,
        apply_subs: bool = False,
    ) -> sp.Expr:
        """
        Polynomial enforcing that AB=CD.

        The condition is expressed through complex directions:
            |z_A-z_B|**2=|z_C-z_D|**2

        """
        expr = self.squared_distance(A, B) - self.squared_distance(C, D)
        return self._apply_all(expr) if apply_subs else expr

    def isogonal_reflection_poly(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        *,
        raw: bool = False,
    ) -> sp.Expr:
        """
        Polynomial enforcing that lines AD and AE are isogonal with respect to angle BAC.

        The condition is expressed through complex directions:

            (z_D - z_A)(z_E - z_A) / conj((z_D - z_A)(z_E - z_A))
            = (z_B - z_A)(z_C - z_A) / conj((z_B - z_A)(z_C - z_A))

        which simplifies to the cross-multiplication below.
        """
        for label in (A, B, C, D, E):
            self.ensure_point(label)

        if B == A or C == A or B == C:
            raise GeometryError(
                "Isogonal reflection requires three distinct vertices A, B, C."
            )

        zA = self.z_symbol(A)
        zB = self.z_symbol(B)
        zC = self.z_symbol(C)
        zD = self.z_symbol(D)
        zE = self.z_symbol(E)

        zbA = self.zb_symbol(A)
        zbB = self.zb_symbol(B)
        zbC = self.zb_symbol(C)
        zbD = self.zb_symbol(D)
        zbE = self.zb_symbol(E)

        left = (zD - zA) * (zE - zA) * (zbB - zbA) * (zbC - zbA)
        right = (zB - zA) * (zC - zA) * (zbD - zbA) * (zbE - zbA)
        expr = left - right
        return expr if raw else sp.expand(expr)

    def isogonal_conjugate_polys(
        self,
        A: str,
        B: str,
        C: str,
        P: str,
        Q: str,
        *,
        raw: bool = False,
    ) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Polynomials enforcing that Q is the isogonal conjugate of P with respect to triangle ABC.

        The condition is captured by requiring AP/AQ, BP/BQ, and CP/CQ pairs to be isogonal at
        each vertex, which reuses the isogonal reflection identity.
        """
        vertex_labels = {A, B, C}
        if len(vertex_labels) != 3:
            raise GeometryError(
                "Isogonal conjugate requires three distinct triangle vertices (A, B, C)."
            )

        for label in (A, B, C, P, Q):
            self.ensure_point(label)

        eq_A = self.isogonal_reflection_poly(A, B, C, P, Q, raw=True)
        eq_B = self.isogonal_reflection_poly(B, C, A, P, Q, raw=True)
        eq_C = self.isogonal_reflection_poly(C, A, B, P, Q, raw=True)

        if raw:
            return eq_A, eq_B, eq_C

        return sp.expand(eq_A), sp.expand(eq_B), sp.expand(eq_C)

    def tangent_circles_poly(self, A: str, B: str, C: str, D: str, E: str, F: str):
        c1_name = self._unique_internal_name("Circ_1")
        c2_name = self._unique_internal_name("Circ_2")

        c1 = self.circle_from_three_points(c1_name, A, B, C)
        c2 = self.circle_from_three_points(c2_name, D, E, F)

        r1_sq = c1.radius_squared
        r2_sq = c2.radius_squared
        d_sq = self.squared_distance(c1.center, c2.center)

        r1_sq = self._apply_all(r1_sq)
        r2_sq = self._apply_all(r2_sq)
        d_sq = self._apply_all(d_sq)
        # print(r1_sq)
        # print(r2_sq)
        # print(d_sq)
        #  (d^2 - R1^2 - R2^2)^2 - 4*R1^2*R2^2 = 0
        tangency_poly = (d_sq - r1_sq - r2_sq) ** 2 - 4 * r1_sq * r2_sq

        return sp.simplify(tangency_poly)

    # ------------------------------------------------------------------
    # Line helpers
    # ------------------------------------------------------------------
    def line_from_coefficients(
        self,
        alpha: sp.Expr,
        beta: sp.Expr,
        gamma: sp.Expr,
        *,
        context: Optional[Sequence[str]] = None,
    ) -> Line:
        """
        Create a Line object from raw coefficients alpha*z + beta*zb + gamma = 0.
        """
        alpha_s = sp.simplify(alpha)
        beta_s = sp.simplify(beta)
        gamma_s = sp.simplify(gamma)
        meta = tuple(context) if context is not None else ()
        return Line(alpha=alpha_s, beta=beta_s, gamma=gamma_s, context=meta)

    def _symbolic_line_label(self, line: Line) -> Optional[str]:
        """Return the registered label for a symbolic line, if available."""
        if not line.context or len(line.context) != 1:
            return None
        label = line.context[0]
        if label in self.line_parameters:
            return label
        return None

    def symbolic_line(self, name: str) -> Line:
        """
        Create a line with fresh symbolic coefficients tied to the provided label.
        """
        alpha, beta, gamma = sp.symbols(
            (
                f"l_{name}_alpha",
                f"l_{name}_beta",
                f"l_{name}_gamma",
            ),
            complex=True,
        )
        self.line_parameters[name] = (alpha, beta, gamma)
        return self.line_from_coefficients(alpha, beta, gamma, context=(name,))

    def line_through_points(self, P: str, Q: str) -> Line:
        """
        Return the line passing through points P and Q.
        """
        self.ensure_point(P)
        self.ensure_point(Q)
        if P == Q:
            raise GeometryError("Line requires two distinct points.")

        zP, zQ = self.z_symbol(P), self.z_symbol(Q)
        zbP, zbQ = self.zb_symbol(P), self.zb_symbol(Q)

        alpha = zbQ - zbP
        beta = -(zQ - zP)
        gamma = (zQ - zP) * zbP - (zbQ - zbP) * zP
        return self.line_from_coefficients(alpha, beta, gamma, context=(P, Q))

    def line_value(self, line: Line, point: str) -> sp.Expr:
        """
        Evaluate the line equation at the specified point label and apply substitutions.
        """
        self.ensure_point(point)
        expr = (
            line.alpha * self.z_symbol(point)
            + line.beta * self.zb_symbol(point)
            + line.gamma
        )
        return self._apply_all(expr)

    def add_point_on_line(self, line: Line, point: str) -> None:
        """
        Store the constraint that the provided point lies on the given line.
        """
        self.ensure_point(point)
        z_val = self.z_symbol(point)
        zb_val = self.zb_symbol(point)
        raw_expr = line.alpha * z_val + line.beta * zb_val + line.gamma

        label = self._symbolic_line_label(line)
        if label is not None:
            alpha, beta, gamma = self.line_parameters[label]
            substitutions_made = False
            for param in (alpha, beta, gamma):
                if param in self.value_subs:
                    raw_expr = sp.simplify(raw_expr.subs(param, self.value_subs[param]))
            for param in (gamma, alpha, beta):
                if param not in self.value_subs and param in raw_expr.free_symbols:
                    solution = sp.solve(raw_expr, param, dict=True)
                    if solution:
                        expr_value = sp.simplify(solution[0][param])
                        self.value_subs[param] = expr_value
                        raw_expr = sp.simplify(raw_expr.subs(param, expr_value))
                        substitutions_made = True
            if substitutions_made:
                raw_expr = sp.simplify(raw_expr)

        expr = sp.expand(raw_expr)
        if expr != 0:
            self.add_constraint(expr)

    def _line_normal_components(self, line: Line) -> Tuple[sp.Expr, sp.Expr]:
        """
        Return the real normal vector components (a, b) corresponding to the line.

        For alpha*z + beta*zb + gamma = 0, the Cartesian coefficients satisfy:
          a = alpha + beta
          b = I * (beta - alpha)
        """
        a = sp.simplify(line.alpha + line.beta)
        b = sp.simplify(sp.I * (line.beta - line.alpha))
        return a, b

    def line_parallel_constraint(self, line1: Line, line2: Line) -> sp.Expr:
        """
        Return the determinant condition enforcing line1 ∥ line2.
        """
        a1, b1 = self._line_normal_components(line1)
        a2, b2 = self._line_normal_components(line2)
        return sp.simplify(a1 * b2 - b1 * a2)

    def line_perpendicular_constraint(self, line1: Line, line2: Line) -> sp.Expr:
        """
        Return the dot-product condition enforcing line1 ⟂ line2.
        """
        a1, b1 = self._line_normal_components(line1)
        a2, b2 = self._line_normal_components(line2)
        return sp.simplify(a1 * a2 + b1 * b2)

    def add_parallel_lines(self, line1: Line, line2: Line) -> None:
        """
        Store the constraint that line1 is parallel to line2.
        """
        constraint = sp.expand(self.line_parallel_constraint(line1, line2))
        self.add_constraint(constraint)

    def add_perpendicular_lines(self, line1: Line, line2: Line) -> None:
        """
        Store the constraint that line1 is perpendicular to line2.
        """

        def _assign_perpendicular_parameters(
            free_line: Line, reference_line: Line
        ) -> None:
            label = self._symbolic_line_label(free_line)
            if label is None:
                return
            alpha, beta, gamma = self.line_parameters[label]
            if alpha in self.value_subs and beta in self.value_subs:
                return

            ref_a, ref_b = self._line_normal_components(reference_line)
            ref_a = self._apply_all(ref_a)
            ref_b = self._apply_all(ref_b)

            # Normal vector perpendicular to (ref_a, ref_b)
            new_a = sp.simplify(-ref_b)
            new_b = sp.simplify(ref_a)

            alpha_expr = sp.simplify((new_a + sp.I * new_b) / 2)
            beta_expr = sp.simplify((new_a - sp.I * new_b) / 2)

            self.value_subs[alpha] = alpha_expr
            self.value_subs[beta] = beta_expr
            if gamma not in self.value_subs:
                # Attempt to determine gamma using any stored point constraint.
                pass

        _assign_perpendicular_parameters(line1, line2)
        _assign_perpendicular_parameters(line2, line1)

        constraint = sp.expand(
            self._apply_all(self.line_perpendicular_constraint(line1, line2))
        )
        if constraint != 0:
            self.add_constraint(constraint)

    def line_intersection(
        self, line1: Line, line2: Line, name: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Construct the intersection point of two stored lines and assign it the provided label.
        """
        self.add_point(name)
        z_var, zb_var = self.z_symbol(name), self.zb_symbol(name)

        eq1 = sp.expand(self._apply_all(line1.evaluate(z_var, zb_var)))
        eq2 = sp.expand(self._apply_all(line2.evaluate(z_var, zb_var)))

        solutions = sp.solve([eq1, eq2], (z_var, zb_var), dict=True)
        if not solutions:
            raise GeometryError(
                "Line intersection failed: lines do not determine a unique point."
            )

        solution = solutions[0]
        self._set_point_assignment(name, solution[z_var], solution[zb_var])
        return self.z(name), self.zb(name)

    # ------------------------------------------------------------------
    # Constraint guards
    # ------------------------------------------------------------------
    def add_collinear(self, A: str, B: str, C: str) -> None:
        """Store the collinearity constraint for (A, B, C)."""
        poly = self.collinear_poly(A, B, C)
        self.add_constraint(poly)

    def add_perpendicular(self, A: str, B: str, C: str, D: str) -> None:
        """Store the perpendicularity constraint AB ⟂ CD."""
        poly = self.perpendicular_poly(A, B, C, D)
        self.add_constraint(poly)

    def add_concyclic(self, A: str, B: str, C: str, D: str) -> None:
        """Store the concyclicity constraint for points A, B, C, D."""
        poly = self.concyclic_poly(A, B, C, D)
        self.add_constraint(poly)

    def add_angle_value(self, A: str, B: str, C: str, angle_radians: sp.Expr) -> None:
        """Store the constraint that angle ABC equals the specified angle (in radians)."""
        poly = self.angle_value_poly(A, B, C, angle_radians)
        self.add_constraint(poly)

    def add_triangle_similarity(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        F: str,
        *,
        directed: bool = True,
    ) -> None:
        """Store similarity constraints between triangles ABC and DEF."""
        for eq in self.triangle_similarity_polys(A, B, C, D, E, F, directed=directed):
            self.add_constraint(eq)

    def add_triangle_congruence(
        self,
        A: str,
        B: str,
        C: str,
        D: str,
        E: str,
        F: str,
        *,
        directed: bool = True,
    ) -> None:
        """Store congruence constraints (triangle equality) between ABC and DEF."""
        for eq in self.triangle_congruence_polys(A, B, C, D, E, F, directed=directed):
            self.add_constraint(eq)

    def add_circumcenter(self, A: str, B: str, C: str, U: str) -> None:
        """
        Store the circumcenter constraints ensuring U is the circumcenter of triangle ABC.
        """
        self.add_point(U)
        eq1, eq2 = self.circumcenter_polys(A, B, C, U)
        self.add_constraint(eq1)
        self.add_constraint(eq2)

    def add_midpoint(self, P: str, Q: str, M: str) -> None:
        """Store the constraint that M is the midpoint of segment PQ."""
        self.add_point(M)
        eq_z, eq_zb = self.midpoint_polys(P, Q, M)
        self.add_constraint(eq_z)
        self.add_constraint(eq_zb)

    def add_centroid_constraint(self, A: str, B: str, C: str, G: str) -> None:
        """Store the constraints declaring G as the centroid of triangle ABC."""
        self.add_point(G)
        eq_z, eq_zb = self.centroid_polys(A, B, C, G)
        self.add_constraint(eq_z)
        self.add_constraint(eq_zb)

    def add_point_reflection(self, P: str, O: str, Q: str) -> None:
        """Store the constraint that Q is the reflection of P across point O."""
        eq_z, eq_zb = self.point_reflection_polys(P, O, Q)
        self.add_constraint(eq_z)
        self.add_constraint(eq_zb)

    def add_line_reflection(self, P: str, A: str, B: str, Q: str) -> None:
        """Store the constraint that Q is the reflection of P across line AB."""
        for eq in self.line_reflection_polys(P, A, B, Q):
            self.add_constraint(eq)

    def add_projection_to_line(self, P: str, A: str, B: str, H: str) -> None:
        """Store the constraints that H is the orthogonal projection of P onto line AB."""
        for eq in self.projection_to_line_polys(P, A, B, H):
            self.add_constraint(eq)

    def add_equal_length(self, A: str, B: str, C: str, D: str):
        """Store the constraints that AB=CD."""
        self.add_constraint(self.equal_length_poly(A, B, C, D))

    def add_equal_angle(self, A: str, B: str, C: str, D: str, E: str, F: str):
        """Store the constraints that 2 directed angle ABC and DEF are equal."""
        self.add_constraint(self.equal_angle_poly(A, B, C, D, E, F))

    def add_isogonal_reflection(self, A: str, B: str, C: str, D: str, E: str) -> None:
        """Store the constraint that AD and AE are isogonal with respect to angle BAC."""
        poly = self.isogonal_reflection_poly(A, B, C, D, E)
        self.add_constraint(poly)

    def add_isogonal_conjugate(self, A: str, B: str, C: str, P: str, Q: str) -> None:
        """Store the constraints that Q is the isogonal conjugate of P with respect to triangle ABC."""
        for eq in self.isogonal_conjugate_polys(A, B, C, P, Q):
            self.add_constraint(eq)

    def isogonal_conjugate_point(
        self, A: str, B: str, C: str, P: str, Q: str
    ) -> sp.Expr:
        """
        Construct the isogonal conjugate Q of point P with respect to triangle ABC by solving
        the angle-bisector reflection equations at two vertices.
        """
        vertex_labels = {A, B, C}
        if len(vertex_labels) != 3:
            raise GeometryError(
                "Isogonal conjugate requires three distinct triangle vertices (A, B, C)."
            )

        for label in (A, B, C, P):
            self.ensure_point(label)
        self.add_point(Q)

        if P in (A, B, C):
            self._set_point_assignment(Q, self.z(P), self.zb(P))
            return self.z(Q)

        eq_A, eq_B, eq_C = self.isogonal_conjugate_polys(A, B, C, P, Q, raw=True)

        zQ, zbQ = self.z_symbol(Q), self.zb_symbol(Q)
        processed_eqs = [self._apply_all(eq_A), self._apply_all(eq_B)]
        solutions = sp.solve(processed_eqs, (zQ, zbQ), dict=True)
        if not solutions:
            raise GeometryError(
                "Isogonal conjugate construction failed: system has no solution."
            )

        eq_C_processed = self._apply_all(eq_C)
        chosen: Optional[Dict[sp.Symbol, sp.Expr]] = None
        for candidate in solutions:
            check = eq_C_processed.subs({zQ: candidate[zQ], zbQ: candidate[zbQ]})
            if sp.simplify(check) == 0:
                chosen = candidate
                break

        if chosen is None:
            chosen = solutions[0]

        self._set_point_assignment(Q, chosen[zQ], chosen[zbQ])

        return self.z(Q)

    def circumcenter(self, A: str, B: str, C: str, U: str) -> sp.Expr:
        """
        Construct the circumcenter U of triangle ABC using the explicit complex coordinate formula.
        """
        self.add_point(U)
        zU, zbU = self.z_symbol(U), self.zb_symbol(U)
        if zU in self.point_assignments and zbU in self.point_assignments:
            return self.z(U)

        a, b, c = self.z(A), self.z(B), self.z(C)
        a_bar, b_bar, c_bar = self.zb(A), self.zb(B), self.zb(C)

        a = self._apply_all(a)
        b = self._apply_all(b)
        c = self._apply_all(c)
        a_bar = self._apply_all(a_bar)
        b_bar = self._apply_all(b_bar)
        c_bar = self._apply_all(c_bar)

        num_z = a * a_bar * (b - c) + b * b_bar * (c - a) + c * c_bar * (a - b)
        den_z = a_bar * (b - c) + b_bar * (c - a) + c_bar * (a - b)

        num_zb = (
            a * a_bar * (b_bar - c_bar)
            + b * b_bar * (c_bar - a_bar)
            + c * c_bar * (a_bar - b_bar)
        )
        den_zb = a * (b_bar - c_bar) + b * (c_bar - a_bar) + c * (a_bar - b_bar)

        den_z = sp.cancel(den_z)
        if den_z == 0:
            raise GeometryError(
                "Circumcenter construction failed: Points A, B, C are collinear (system is underdetermined)."
            )

        den_zb = sp.cancel(den_zb)

        z_val = sp.cancel(num_z / den_z)
        zb_val = sp.cancel(num_zb / den_zb)

        self._set_point_assignment(U, z_val, zb_val)
        return self.z(U)

    def midpoint(self, P: str, Q: str, M: str) -> sp.Expr:
        """
        Construct the midpoint M of segment PQ.
        """
        self.add_point(M)
        z_expr = sp.simplify((self.z(P) + self.z(Q)) / 2)
        zb_expr = sp.simplify((self.zb(P) + self.zb(Q)) / 2)
        self._set_point_assignment(M, z_expr, zb_expr)
        return self.z(M)

    def euler_center(
        self,
        A: str,
        B: str,
        C: str,
        N: str,
        circumcenter_name: Optional[str] = None,
        orthocenter_name: Optional[str] = None,
    ) -> sp.Expr:
        """
        Construct the Euler (nine-point) center N as midpoint of circumcenter and orthocenter.

        Optional circumcenter/orthocenter labels allow callers to reuse existing points; otherwise
        temporary internal names are introduced for the construction.
        """
        circ_label = circumcenter_name or self._unique_internal_name("EulerCirc")
        orth_label = orthocenter_name or self._unique_internal_name("EulerOrth")

        self.circumcenter(A, B, C, circ_label)
        self.orthocenter_via_altitudes(A, B, C, orth_label)
        self.add_point(N)

        z_expr = sp.simplify((self.z(circ_label) + self.z(orth_label)) / 2)
        zb_expr = sp.simplify((self.zb(circ_label) + self.zb(orth_label)) / 2)
        self._set_point_assignment(N, z_expr, zb_expr)
        return self.z(N)

    def lemoine_point(
        self,
        A: str,
        B: str,
        C: str,
        L: str,
        circumcenter_name: Optional[str] = None,
    ) -> sp.Expr:
        """
        Construct the Lemoine (symmedian) point via intersections of circumcircle tangents.
        """
        circ_label = circumcenter_name or self._unique_internal_name("LemoineCirc")

        self.circumcenter(A, B, C, circ_label)

        tangent_BC = self._unique_internal_name(f"T_{B}{C}")
        self._tangent_intersection(B, C, circ_label, tangent_BC)

        tangent_CA = self._unique_internal_name(f"T_{C}{A}")
        self._tangent_intersection(C, A, circ_label, tangent_CA)

        self.intersection_of_lines(A, tangent_BC, B, tangent_CA, L)
        return self.z(L)

    def reflect_point_over_point(self, P: str, O: str, Q: str) -> sp.Expr:
        """
        Construct the reflection of point P across point O and assign it to Q.
        """
        self.add_point(Q)
        z_expr = sp.simplify(2 * self.z(O) - self.z(P))
        zb_expr = sp.simplify(2 * self.zb(O) - self.zb(P))
        self._set_point_assignment(Q, z_expr, zb_expr)
        return self.z(Q)

    def reflect_point_over_line(self, P: str, A: str, B: str, Q: str) -> sp.Expr:
        """
        Construct the reflection of point P across line AB and assign it to Q.
        """
        self.add_point(Q)
        midpoint_label = self._midpoint_label(P, Q)
        self.add_point(midpoint_label)

        eq_perp, eq_mid_z, eq_mid_zb, eq_collinear = self.line_reflection_polys(
            P, A, B, Q, raw=True
        )
        equations = [eq_perp, eq_mid_z, eq_mid_zb, eq_collinear]

        zQ, zbQ = self.z_symbol(Q), self.zb_symbol(Q)
        zM, zbM = self.z_symbol(midpoint_label), self.zb_symbol(midpoint_label)
        processed = [self._apply_all(eq) for eq in equations]
        solutions = sp.solve(processed, (zQ, zbQ, zM, zbM), dict=True)
        if not solutions:
            raise GeometryError(
                "Line reflection construction failed: system has no solution."
            )

        solution = solutions[0]
        self._set_point_assignment(Q, solution[zQ], solution[zbQ])
        self._set_point_assignment(midpoint_label, solution[zM], solution[zbM])
        return self.z(Q)

    # ------------------------------------------------------------------
    # Constructions
    # ------------------------------------------------------------------
    def set_main_unit_triangle(
        self,
        A: str,
        B: str,
        C: str,
        *,
        root_names: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Declare triangle ABC as the main unit-circle triangle and assign square-root auxiliaries.

        Each vertex receives a dedicated auxiliary point X such that z_A = z_X^2, with X constrained
        to the unit circle.  The configuration is stored for downstream helpers (incenter, excenters,
        arc midpoints).  Optional `root_names` supplies custom labels (x, y, z) for the auxiliary
        square-root points corresponding to A, B, C respectively.
        """
        canonical_labels = ("A", "B", "C")
        points_map = dict(zip(canonical_labels, (A, B, C)))
        if len({A, B, C}) != 3:
            raise GeometryError(
                "Main unit triangle requires three distinct vertex labels."
            )

        for vertex in points_map.values():
            self.ensure_point(vertex)
            if vertex not in self.unit_circle_points:
                raise GeometryError(
                    f"Point '{vertex}' must be declared on the unit circle before configuring the main triangle."
                )

        if root_names is not None:
            if len(root_names) != 3:
                raise GeometryError(
                    "root_names must provide exactly three auxiliary labels."
                )
            if any(not isinstance(label, str) for label in root_names):
                raise GeometryError("All root_names entries must be strings.")
            if len(set(root_names)) != 3:
                raise GeometryError("root_names must contain three distinct labels.")
            if any(label in points_map.values() for label in root_names):
                raise GeometryError(
                    "root_names must differ from the main triangle vertex labels."
                )
            root_sequence = tuple(root_names)
        else:
            root_sequence = (
                self._unit_root_name("A"),
                self._unit_root_name("B"),
                self._unit_root_name("C"),
            )

        roots: Dict[str, str] = {}
        for index, canonical in enumerate(canonical_labels):
            vertex = points_map[canonical]
            root_label = root_sequence[index]
            self.add_point(root_label)
            self.add_unit_circle(root_label)
            z_root = self.z_symbol(root_label)
            zb_root = self.zb_symbol(root_label)
            self._set_point_assignment(
                vertex, sp.simplify(z_root**2), sp.simplify(zb_root**2)
            )
            self.model_subs[self.zb_symbol(vertex)] = 1 / sp.simplify((z_root**2))
            roots[canonical] = root_label

        self._main_unit_triangle = UnitTriangleConfig(points=points_map, roots=roots)

    def main_triangle_incenter(self, name: str) -> sp.Expr:
        """
        Assign the incenter of the configured main triangle using the -xy - yz - zx formula.
        """
        config = self._require_main_unit_triangle()
        self.add_point(name)

        rootA = config.roots["A"]
        rootB = config.roots["B"]
        rootC = config.roots["C"]

        z_rootA = self.z_symbol(rootA)
        z_rootB = self.z_symbol(rootB)
        z_rootC = self.z_symbol(rootC)
        zb_rootA = self.zb_symbol(rootA)
        zb_rootB = self.zb_symbol(rootB)
        zb_rootC = self.zb_symbol(rootC)

        z_expr = -(z_rootA * z_rootB + z_rootB * z_rootC + z_rootC * z_rootA)
        zb_expr = -(zb_rootA * zb_rootB + zb_rootB * zb_rootC + zb_rootC * zb_rootA)
        self._set_point_assignment(name, sp.simplify(z_expr), sp.simplify(zb_expr))
        return self.z(name)

    def main_triangle_excenter(self, which: str, name: str) -> sp.Expr:
        """
        Assign the specified excenter of the configured main triangle using xy±yz±zx formulas.

        Parameters
        ----------
        which:
            One of 'A', 'B', 'C', selecting the excenter opposite the corresponding vertex.
        name:
            Label for the constructed excenter.
        """
        config = self._require_main_unit_triangle()
        normalized = which.upper()
        if normalized not in config.roots:
            raise GeometryError(
                f"Excenter '{which}' is not valid for the configured main triangle."
            )

        rootA = config.roots["A"]
        rootB = config.roots["B"]
        rootC = config.roots["C"]
        z_rootA = self.z_symbol(rootA)
        z_rootB = self.z_symbol(rootB)
        z_rootC = self.z_symbol(rootC)
        zb_rootA = self.zb_symbol(rootA)
        zb_rootB = self.zb_symbol(rootB)
        zb_rootC = self.zb_symbol(rootC)

        patterns = {
            "A": (sp.Integer(1), sp.Integer(-1), sp.Integer(1)),
            "B": (sp.Integer(1), sp.Integer(1), sp.Integer(-1)),
            "C": (sp.Integer(-1), sp.Integer(1), sp.Integer(1)),
        }
        coeff_ab, coeff_bc, coeff_ca = patterns[normalized]

        z_expr = (
            coeff_ab * z_rootA * z_rootB
            + coeff_bc * z_rootB * z_rootC
            + coeff_ca * z_rootC * z_rootA
        )
        zb_expr = (
            coeff_ab * zb_rootA * zb_rootB
            + coeff_bc * zb_rootB * zb_rootC
            + coeff_ca * zb_rootC * zb_rootA
        )

        self.add_point(name)
        self._set_point_assignment(name, sp.simplify(z_expr), sp.simplify(zb_expr))
        return self.z(name)

    def main_triangle_arc_midpoint(
        self, which: str, name: str, *, containing_vertex: bool = False
    ) -> sp.Expr:
        """
        Assign the midpoint of the arc opposite vertex `which` on the circumcircle.

        Parameters
        ----------
        which:
            One of 'A', 'B', 'C'.  For example, 'A' yields the midpoint of arc BC.
        name:
            Label for the constructed midpoint.
        containing_vertex:
            If False (default), return the midpoint of the arc not containing the vertex.
            If True, return the midpoint of the complementary arc containing the vertex.
        """
        config = self._require_main_unit_triangle()
        normalized = which.upper()
        if normalized not in config.roots:
            raise GeometryError(
                f"Arc midpoint '{which}' is not valid for the configured main triangle."
            )

        if normalized == "A":
            root1, root2 = config.roots["B"], config.roots["C"]
        elif normalized == "B":
            root1, root2 = config.roots["C"], config.roots["A"]
        else:  # normalized == "C"
            root1, root2 = config.roots["A"], config.roots["B"]

        sign = sp.Integer(1) if containing_vertex else sp.Integer(-1)
        z_expr = sign * self.z_symbol(root1) * self.z_symbol(root2)
        zb_expr = sign * self.zb_symbol(root1) * self.zb_symbol(root2)

        self.add_point(name)
        self._set_point_assignment(name, sp.simplify(z_expr), sp.simplify(zb_expr))
        self.add_unit_circle(name)
        return self.z(name)

    def orthocenter_via_altitudes(
        self, T1: str, T2: str, T3: str, H: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Construct the orthocenter H of triangle T1T2T3 by solving the two altitude constraints.
        """
        self.add_point(H)
        zH, zbH = self.z_symbol(H), self.zb_symbol(H)

        eq1 = self._apply_all(self.perpendicular_poly(T1, H, T2, T3))
        eq2 = self._apply_all(self.perpendicular_poly(T2, H, T1, T3))
        solutions = sp.solve([eq1, eq2], (zH, zbH), dict=True)
        if not solutions:
            raise GeometryError(
                "Orthocenter construction failed: system is underdetermined."
            )

        solution = solutions[0]
        self._set_point_assignment(H, solution[zH], solution[zbH])
        return self.z(H), self.zb(H)

    def intersection_of_lines(
        self, A: str, B: str, C: str, D: str, X: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Intersection point X of lines AB and CD.
        """
        self.add_point(X)
        zX, zbX = self.z_symbol(X), self.zb_symbol(X)

        eq1 = self._apply_all(self.collinear_poly(X, A, B))
        eq2 = self._apply_all(self.collinear_poly(X, C, D))
        solutions = sp.solve([eq1, eq2], (zX, zbX), dict=True)
        if not solutions:
            raise GeometryError(
                "Intersection construction failed: lines do not determine a unique point."
            )

        solution = solutions[0]
        self._set_point_assignment(X, solution[zX], solution[zbX])
        return self.z(X), self.zb(X)

    def calculate_point(self, name: str) -> Tuple[sp.Expr, sp.Expr]:
        """
        Explicitly derive the coordinates of ``name`` from stored constraints.

        The calculation keeps the ordinary coordinate as the primary variable:
        it first derives ``zb_name`` as an expression in ``z_name`` and known
        data, then substitutes that relation to determine ``z_name`` when the
        remaining constraints are sufficient.  A line constraint therefore
        records a useful partial assignment, while two independent constraints
        can fully resolve the point.
        """
        self.ensure_point(name)
        z_var = self.z_symbol(name)
        zb_var = self.zb_symbol(name)
        relevant_raw = [
            constraint
            for constraint in self.constraints
            if constraint.has(z_var, zb_var)
        ]
        if not relevant_raw:
            raise GeometryError(
                f"Cannot calculate point '{name}': no stored constraint references it."
            )

        equations = [
            sp.expand(self._apply_all(constraint)) for constraint in relevant_raw
        ]
        equations = [equation for equation in equations if equation != 0]
        if not equations:
            return self.z(name), self.zb(name)

        zb_candidates: List[sp.Expr] = []
        for equation in equations:
            try:
                solutions = sp.solve(equation, zb_var, dict=True)
            except (NotImplementedError, ValueError):
                continue
            for solution in solutions:
                if zb_var not in solution:
                    continue
                candidate = sp.simplify(self._apply_all(solution[zb_var]))
                if zb_var not in candidate.free_symbols:
                    zb_candidates.append(candidate)

        if not zb_candidates:
            raise GeometryError(
                f"Cannot calculate point '{name}': unable to derive its conjugate coordinate."
            )

        zb_expr = min(zb_candidates, key=sp.count_ops)
        self._set_coordinate_assignment(zb_var, zb_expr)
        reduced_equations = [
            sp.expand(self._apply_all(constraint)) for constraint in relevant_raw
        ]
        reduced_equations = [
            equation
            for equation in reduced_equations
            if equation != 0 and z_var in equation.free_symbols
        ]
        if not reduced_equations:
            return self.z(name), self.zb(name)
        try:
            z_solutions = sp.solve(reduced_equations, z_var, dict=True)
        except (NotImplementedError, ValueError) as exc:
            raise GeometryError(
                f"Cannot calculate point '{name}': failed to solve its ordinary coordinate."
            ) from exc

        resolved_z = []
        for solution in z_solutions:
            if z_var not in solution:
                continue
            candidate = sp.simplify(self._apply_all(solution[z_var]))
            if (
                z_var not in candidate.free_symbols
                and zb_var not in candidate.free_symbols
            ):
                if not any(
                    sp.simplify(candidate - existing) == 0 for existing in resolved_z
                ):
                    resolved_z.append(candidate)

        if not resolved_z:
            raise GeometryError(
                f"Cannot calculate point '{name}': constraints do not determine its ordinary coordinate."
            )
        if len(resolved_z) != 1:
            raise GeometryError(
                f"Cannot calculate point '{name}': constraints produce multiple solutions."
            )

        self._set_coordinate_assignment(z_var, resolved_z[0])
        self._set_coordinate_assignment(zb_var, self._apply_all(zb_expr))
        return self.z(name), self.zb(name)

    def project_point_to_line(
        self, P: str, A: str, B: str, H: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Construct the orthogonal projection H of point P onto line AB.
        """
        if A == B:
            raise GeometryError(
                "Projection requires two distinct points to define line AB."
            )

        for label in (P, A, B):
            self.ensure_point(label)

        self.add_point(H)
        if self._has_assignment(H):
            return self.z(H), self.zb(H)

        zH, zbH = self.z_symbol(H), self.zb_symbol(H)
        eq_perp, eq_collinear = self.projection_to_line_polys(P, A, B, H, raw=True)
        candidates = self._solve_candidate_points(zH, zbH, [eq_perp, eq_collinear])
        if not candidates:
            raise GeometryError(
                f"Projection failed: line '{A}{B}' and point '{P}' do not determine a unique foot."
            )

        z_value, zb_value = candidates[0]
        self._set_point_assignment(H, z_value, zb_value)
        return self.z(H), self.zb(H)

    def line_circle_intersection(
        self,
        line_point1: str,
        line_point2: str,
        center: str,
        radius_point: str,
        name: str,
        *,
        avoid: Optional[Sequence[str]] = None,
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Construct the intersection of line (line_point1 line_point2) with the circle centered at `center`
        passing through `radius_point`.

        Parameters
        ----------
        line_point1, line_point2:
            Points defining the target line.
        center:
            Circle center.
        radius_point:
            Point fixing the circle radius (must be distinct from center).
        name:
            Label to assign the constructed intersection.
        avoid:
            Optional list of point labels to skip if multiple intersections exist.
        """
        self.add_point(name)
        self.ensure_point(line_point1)
        self.ensure_point(line_point2)
        circle = self._circle_from_center_radius_point(center, radius_point)
        z_var, zb_var = self.z_symbol(name), self.zb_symbol(name)

        eq_line = self.collinear_poly(name, line_point1, line_point2, raw=True)
        eq_circle = circle.evaluate(self, z_var, zb_var)

        avoid_pairs = self._prepare_avoid_pairs(avoid)
        candidates = self._solve_candidate_points(
            z_var, zb_var, [eq_line, eq_circle], avoid_pairs=avoid_pairs
        )
        if not candidates:
            raise GeometryError(
                f"Line through '{line_point1}{line_point2}' does not meet the circle centered at '{center}'."
            )

        z_value, zb_value = candidates[0]
        self._set_point_assignment(name, z_value, zb_value)
        return self.z(name), self.zb(name)

    def circle_intersection(
        self,
        center1: str,
        radius_point1: str,
        center2: str,
        radius_point2: str,
        name: str,
        *,
        avoid: Optional[Sequence[str]] = None,
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Construct the intersection of the two circles (center1, radius_point1) and (center2, radius_point2).
        """
        self.add_point(name)
        z_var, zb_var = self.z_symbol(name), self.zb_symbol(name)

        circle1 = self._circle_from_center_radius_point(center1, radius_point1)
        circle2 = self._circle_from_center_radius_point(center2, radius_point2)

        eq_circle1 = circle1.evaluate(self, z_var, zb_var)
        eq_circle2 = circle2.evaluate(self, z_var, zb_var)

        avoid_pairs = self._prepare_avoid_pairs(avoid)
        candidates = self._solve_candidate_points(
            z_var, zb_var, [eq_circle1, eq_circle2], avoid_pairs=avoid_pairs
        )
        if not candidates:
            raise GeometryError(
                f"Circles '{center1}' and '{center2}' do not intersect."
            )

        z_value, zb_value = candidates[0]
        self._set_point_assignment(name, z_value, zb_value)
        return self.z(name), self.zb(name)

    def line_circle_object_intersections(
        self,
        line_name: str,
        circle_name: str,
        point_names: Sequence[str],
        *,
        avoid: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Assign intersection points between a stored line and circle to the provided labels.
        """
        if line_name not in self.lines:
            raise GeometryError(f"Line '{line_name}' is not defined.")
        if circle_name not in self.circles:
            raise GeometryError(f"Circle '{circle_name}' is not defined.")
        ordered_names = self._normalize_label_sequence(
            list(point_names), "Intersection point"
        )

        line = self.lines[line_name]
        circle = self.circles[circle_name]

        z_var = sp.Symbol(f"__line_circle_{line_name}_{circle_name}_z")
        zb_var = sp.Symbol(f"__line_circle_{line_name}_{circle_name}_zb")

        eq_line = line.evaluate(z_var, zb_var)
        eq_circle = circle.evaluate(self, z_var, zb_var)

        avoid_pairs = self._prepare_avoid_pairs(avoid)
        candidates = self._solve_candidate_points(
            z_var, zb_var, [eq_line, eq_circle], avoid_pairs=avoid_pairs
        )
        if not candidates:
            raise GeometryError(
                f"Line '{line_name}' and circle '{circle_name}' do not intersect."
            )
        if len(ordered_names) > len(candidates):
            raise GeometryError(
                "Requested more intersection points than available solutions."
            )

        for name, (z_value, zb_value) in zip(ordered_names, candidates):
            self.add_point(name)
            self._set_point_assignment(name, z_value, zb_value)

    def circle_object_intersections(
        self,
        circle1_name: str,
        circle2_name: str,
        point_names: Sequence[str],
        *,
        avoid: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Assign intersection points between two stored circles to the provided labels.
        """
        if circle1_name not in self.circles:
            raise GeometryError(f"Circle '{circle1_name}' is not defined.")
        if circle2_name not in self.circles:
            raise GeometryError(f"Circle '{circle2_name}' is not defined.")
        ordered_names = self._normalize_label_sequence(
            list(point_names), "Intersection point"
        )

        circle1 = self.circles[circle1_name]
        circle2 = self.circles[circle2_name]

        z_var = sp.Symbol(f"__circle_circle_{circle1_name}_{circle2_name}_z")
        zb_var = sp.Symbol(f"__circle_circle_{circle1_name}_{circle2_name}_zb")

        eq_circle1 = circle1.evaluate(self, z_var, zb_var)
        eq_circle2 = circle2.evaluate(self, z_var, zb_var)

        avoid_pairs = self._prepare_avoid_pairs(avoid)
        candidates = self._solve_candidate_points(
            z_var, zb_var, [eq_circle1, eq_circle2], avoid_pairs=avoid_pairs
        )
        if not candidates:
            raise GeometryError(
                f"Circles '{circle1_name}' and '{circle2_name}' do not intersect."
            )
        if len(ordered_names) > len(candidates):
            raise GeometryError(
                "Requested more intersection points than available solutions."
            )

        for name, (z_value, zb_value) in zip(ordered_names, candidates):
            self.add_point(name)
            self._set_point_assignment(name, z_value, zb_value)

    def radical_line_from_circles(
        self, circle1_name: str, circle2_name: str, line_name: str
    ) -> Line:
        """Construct the radical line of two stored circles and register it under `line_name`."""
        if circle1_name not in self.circles or circle2_name not in self.circles:
            missing = [
                name
                for name in (circle1_name, circle2_name)
                if name not in self.circles
            ]
            raise GeometryError(f"Circles not defined: {', '.join(missing)}.")
        if line_name in self.lines:
            raise GeometryError(f"Line '{line_name}' is already defined.")

        circle1 = self.circles[circle1_name]
        circle2 = self.circles[circle2_name]

        z_c1 = self._apply_all(self.z(circle1.center))
        zb_c1 = self._apply_all(self.zb(circle1.center))
        z_c2 = self._apply_all(self.z(circle2.center))
        zb_c2 = self._apply_all(self.zb(circle2.center))
        r1_sq = self._apply_all(circle1.radius_squared)
        r2_sq = self._apply_all(circle2.radius_squared)

        alpha = sp.simplify(zb_c2 - zb_c1)
        beta = sp.simplify(z_c2 - z_c1)
        gamma = sp.simplify(z_c1 * zb_c1 - r1_sq - z_c2 * zb_c2 + r2_sq)

        if alpha == 0 and beta == 0:
            raise GeometryError("Radical line is undefined for coincident circles.")

        line = self.line_from_coefficients(
            alpha, beta, gamma, context=(line_name, circle1_name, circle2_name)
        )
        self.lines[line_name] = line
        return line

    def tangent_lines_from_point_to_circle(
        self,
        point: str,
        circle_name: str,
        line_names: Sequence[str],
        *,
        tangent_point_names: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Construct tangent lines from `point` to the stored circle and assign them to the provided labels.

        Returns two lines when the point lies outside the circle, a single line when the point lies on
        the circle, and raises a GeometryError if the point lies strictly inside the circle.
        """
        if circle_name not in self.circles:
            raise GeometryError(f"Circle '{circle_name}' is not defined.")
        self.ensure_point(point)
        ordered_lines = self._normalize_label_sequence(list(line_names), "Line")
        if not ordered_lines:
            raise GeometryError(
                "At least one line label must be provided for tangent construction."
            )
        for name in ordered_lines:
            if name in self.lines:
                raise GeometryError(f"Line '{name}' is already defined.")

        circle = self.circles[circle_name]
        z_center = self._apply_all(self.z(circle.center))
        zb_center = self._apply_all(self.zb(circle.center))
        z_point = self._apply_all(self.z(point))
        zb_point = self._apply_all(self.zb(point))

        position = self._classify_point_against_circle(point, circle)
        if position == "inside":
            raise GeometryError(
                f"Point '{point}' lies inside circle '{circle_name}'; tangents do not exist."
            )

        z_var = sp.Symbol(f"__tangent_{point}_{circle_name}_z")
        zb_var = sp.Symbol(f"__tangent_{point}_{circle_name}_zb")

        eq_circle = self._apply_all(circle.evaluate(self, z_var, zb_var))
        eq_perp = self._apply_all(
            (z_var - z_center) * (zb_point - zb_var)
            + (zb_var - zb_center) * (z_point - z_var)
        )

        candidates = self._solve_candidate_points(z_var, zb_var, [eq_circle, eq_perp])
        if not candidates:
            raise GeometryError(
                f"No tangents found from '{point}' to circle '{circle_name}'."
            )

        if position == "on":
            candidates = candidates[:1]
        elif position == "outside":
            if len(candidates) < 2:
                raise GeometryError(
                    f"Expected two tangents from '{point}' to circle '{circle_name}', but solver returned fewer."
                )
            candidates = candidates[:2]
        else:  # unknown classification
            candidates = candidates[: len(ordered_lines)]

        if len(ordered_lines) != len(candidates):
            raise GeometryError(
                f"Expected {len(candidates)} line label(s) for the tangents but received {len(ordered_lines)}."
            )

        tangent_labels: Optional[List[str]] = None
        if tangent_point_names is not None:
            tangent_labels = self._normalize_label_sequence(
                list(tangent_point_names), "Tangent point"
            )
            if len(tangent_labels) != len(candidates):
                raise GeometryError(
                    "Tangent point labels must match the number of tangent lines."
                )

        for index, (line_name, (z_tangent, zb_tangent)) in enumerate(
            zip(ordered_lines, candidates)
        ):
            if tangent_labels is not None:
                tangent_point_label = tangent_labels[index]
            else:
                tangent_point_label = self._unique_internal_name(
                    f"tangent_point_{point}_{circle_name}_{index}"
                )

            self.add_point(tangent_point_label)
            self._set_point_assignment(tangent_point_label, z_tangent, zb_tangent)

            alpha = sp.simplify(zb_tangent - zb_center)
            beta = sp.simplify(z_tangent - z_center)
            gamma = sp.simplify(-alpha * z_tangent - beta * zb_tangent)

            line = self.line_from_coefficients(
                alpha, beta, gamma, context=(point, circle_name, tangent_point_label)
            )
            self.lines[line_name] = line

    def centroid(self, A: str, B: str, C: str, G: str) -> sp.Expr:
        """
        Assign centroid G of triangle ABC.
        """
        self.add_point(G)
        z_expr = sp.simplify((self.z(A) + self.z(B) + self.z(C)) / 3)
        zb_expr = sp.simplify((self.zb(A) + self.zb(B) + self.zb(C)) / 3)
        self._set_point_assignment(G, z_expr, zb_expr)
        return self.z(G)

    def squared_distance(self, P: str, Q: str) -> sp.Expr:
        """
        Return the squared distance |P - Q|^2 using current substitutions.
        """
        self.ensure_point(P)
        self.ensure_point(Q)
        diff_z = sp.simplify(self.z(P) - self.z(Q))
        diff_zb = sp.simplify(self.zb(P) - self.zb(Q))
        return sp.simplify(diff_z * diff_zb)

    def _set_point_assignment(
        self, name: str, z_expr: sp.Expr, zb_expr: sp.Expr
    ) -> None:
        """Register rational expressions for a constructed point."""
        z_candidate = sp.simplify(sp.together(z_expr))
        zb_candidate = sp.simplify(sp.together(zb_expr))
        self._enforce_distinct_candidate(name, z_candidate, zb_candidate)
        z_symbol = self.z_symbol(name)
        zb_symbol = self.zb_symbol(name)
        self.point_assignments[z_symbol] = z_candidate
        self.point_assignments[zb_symbol] = zb_candidate

    def _set_coordinate_assignment(
        self, symbol: sp.Symbol, expression: sp.Expr
    ) -> None:
        """Record one coordinate assignment, including an intentional partial one."""
        self.point_assignments[symbol] = sp.simplify(sp.together(expression))

    def _fermat_point_components(
        self, A: str, B: str, C: str, w: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Return (z, zb) expressions for the Fermat point of triangle ABC associated with root `w`.
        """
        for label in (A, B, C):
            self.ensure_point(label)

        zA, zB, zC = self.z(A), self.z(B), self.z(C)
        zbA, zbB, zbC = self.zb(A), self.zb(B), self.zb(C)

        conj_bc = sp.simplify(zbB - zbC)
        conj_ca = sp.simplify(zbC - zbA)
        conj_ab = sp.simplify(zbA - zbB)

        w_sq = sp.simplify(w**2)

        numerator = zA * conj_bc + w * zB * conj_ca + w_sq * zC * conj_ab
        denominator = sp.simplify(conj_bc + w * conj_ca + w_sq * conj_ab)
        if denominator == 0:
            raise GeometryError(
                "Fermat point construction is undefined: denominator vanished."
            )

        diff_bc = sp.simplify(zB - zC)
        diff_ca = sp.simplify(zC - zA)
        diff_ab = sp.simplify(zA - zB)

        w_conj = sp.conjugate(w)
        w_conj_sq = sp.conjugate(w_sq)

        numerator_conj = (
            zbA * diff_bc + w_conj * zbB * diff_ca + w_conj_sq * zbC * diff_ab
        )
        denominator_conj = sp.simplify(diff_bc + w_conj * diff_ca + w_conj_sq * diff_ab)
        if denominator_conj == 0:
            raise GeometryError(
                "Fermat point construction is undefined: conjugate denominator vanished."
            )

        z_expr = sp.simplify(numerator / denominator)
        zb_expr = sp.simplify(numerator_conj / denominator_conj)
        return z_expr, zb_expr

    def add_fermat_points(
        self, A: str, B: str, C: str, F1: str, F2: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Assign the two Fermat points of triangle ABC to the provided labels.
        """
        if len({A, B, C}) != 3:
            raise GeometryError(
                "Fermat point construction expects three distinct vertices."
            )
        if F1 == F2:
            raise GeometryError("Fermat point labels must be distinct.")

        roots = (sp.exp(2 * sp.pi * sp.I / 3), sp.exp(4 * sp.pi * sp.I / 3))
        outputs: List[sp.Expr] = []
        for label, root in zip((F1, F2), roots):
            if not isinstance(label, str):
                raise GeometryError("Fermat point labels must be strings.")
            self.add_point(label)
            z_expr, zb_expr = self._fermat_point_components(A, B, C, root)
            self._set_point_assignment(label, z_expr, zb_expr)
            outputs.append(self.z(label))
        return outputs[0], outputs[1]

    def _unique_internal_name(self, base: str) -> str:
        """Return a point name unlikely to clash with user-provided labels."""
        sanitized = base.replace(" ", "_")
        candidate = f"__{sanitized}"
        index = 1
        while candidate in self.points:
            candidate = f"__{sanitized}_{index}"
            index += 1
        return candidate

    def _unit_root_name(self, vertex: str) -> str:
        """Deterministic label for the auxiliary square-root point of a vertex."""
        sanitized = vertex.replace(" ", "_")
        return f"__unit_root_{sanitized}"

    def _midpoint_label(self, P: str, Q: str) -> str:
        """Deterministic internal name for the midpoint of segment PQ."""
        ordered = tuple(sorted((P, Q)))
        return f"__mid_{ordered[0]}_{ordered[1]}"

    def _normalize_label_sequence(
        self,
        labels: Sequence[Any],
        kind: str,
        *,
        allow_empty: bool = False,
    ) -> List[str]:
        """Validate and normalize a sequence of labels."""
        if not isinstance(labels, (list, tuple)):
            raise GeometryError(f"{kind} labels must be provided as a list.")
        if not labels:
            if allow_empty:
                return []
            raise GeometryError(f"{kind} labels must be provided.")

        normalized: List[str] = []
        for label in labels:
            if not isinstance(label, str):
                raise GeometryError(f"{kind} labels must be strings.")
            if label in normalized:
                raise GeometryError(f"{kind} labels must be unique.")
            normalized.append(label)
        return normalized

    def _prepare_avoid_pairs(
        self, avoid: Optional[Sequence[str]]
    ) -> List[Tuple[sp.Expr, sp.Expr]]:
        """Convert avoid label list into coordinate pairs."""
        if avoid is None:
            return []
        if isinstance(avoid, str):
            avoid = [avoid]
        if not isinstance(avoid, (list, tuple)):
            raise GeometryError("avoid must be a string or list of point labels.")

        pairs: List[Tuple[sp.Expr, sp.Expr]] = []
        for label in avoid:
            if not isinstance(label, str):
                raise GeometryError("avoid entries must be point labels.")
            self.ensure_point(label)
            pairs.append((self.z(label), self.zb(label)))
        return pairs

    def _solve_candidate_points(
        self,
        z_var: sp.Symbol,
        zb_var: sp.Symbol,
        equations: Sequence[sp.Expr],
        *,
        avoid_pairs: Optional[Sequence[Tuple[sp.Expr, sp.Expr]]] = None,
    ) -> List[Tuple[sp.Expr, sp.Expr]]:
        """Solve a system and return distinct candidate (z, zb) pairs respecting avoid constraints."""
        processed = [self._apply_all(eq) for eq in equations]
        solutions = sp.solve(processed, (z_var, zb_var), dict=True)
        if not solutions:
            return []

        candidates: List[Tuple[sp.Expr, sp.Expr]] = []
        avoid_pairs = list(avoid_pairs or [])

        for solution in solutions:
            z_candidate = sp.simplify(solution[z_var])
            zb_candidate = sp.simplify(solution[zb_var])

            duplicate = False
            for existing_z, existing_zb in candidates:
                if (
                    sp.simplify(z_candidate - existing_z) == 0
                    and sp.simplify(zb_candidate - existing_zb) == 0
                ):
                    duplicate = True
                    break
            if duplicate:
                continue

            skip = False
            for z_avoid, zb_avoid in avoid_pairs:
                if (
                    sp.simplify(z_candidate - z_avoid) == 0
                    and sp.simplify(zb_candidate - zb_avoid) == 0
                ):
                    skip = True
                    break
            if skip:
                continue

            candidates.append((z_candidate, zb_candidate))
        return candidates

    def _classify_point_against_circle(self, point: str, circle: Circle) -> str:
        """Return a heuristic classification of the point relative to the circle."""
        delta_expr = sp.simplify(
            self._apply_all(
                self.squared_distance(point, circle.center) - circle.radius_squared
            )
        )
        if delta_expr == 0 or getattr(delta_expr, "is_zero", False):
            return "on"
        if getattr(delta_expr, "is_negative", False):
            return "inside"
        if getattr(delta_expr, "is_positive", False):
            return "outside"
        return "unknown"

    def _has_assignment(self, name: str) -> bool:
        """Return True if both coordinates of the point have recorded assignments."""
        if name not in self.points:
            return False
        record = self.points[name]
        return (
            record.z in self.point_assignments and record.zb in self.point_assignments
        )

    def _circle_from_center_radius_point(
        self, center: str, radius_point: str
    ) -> Circle:
        """Instantiate a circle determined by a given center and point on the circle."""
        self.ensure_point(center)
        self.ensure_point(radius_point)
        radius_sq = self.squared_distance(center, radius_point)
        return Circle(
            center=center,
            radius_squared=sp.simplify(radius_sq),
            context=(center, radius_point),
        )

    def _validate_distinct_pair(self, P: str, Q: str) -> None:
        """Ensure the distinct constraint between P and Q is not currently violated."""
        if self._has_assignment(P) and self._has_assignment(Q):
            zP = self.point_assignments[self.points[P].z]
            zbP = self.point_assignments[self.points[P].zb]
            zQ = self.point_assignments[self.points[Q].z]
            zbQ = self.point_assignments[self.points[Q].zb]
            if sp.simplify(zP - zQ) == 0 and sp.simplify(zbP - zbQ) == 0:
                raise GeometryError(
                    f"Distinct constraint violated: points '{P}' and '{Q}' coincide."
                )

    def _enforce_distinct_candidate(
        self, name: str, z_candidate: sp.Expr, zb_candidate: sp.Expr
    ) -> None:
        """Verify that assigning `name` to the candidate coordinates preserves distinct constraints."""
        relevant_pairs = [pair for pair in self.distinct_pairs if name in pair]
        if not relevant_pairs:
            return
        simplified_z = sp.simplify(z_candidate)
        simplified_zb = sp.simplify(zb_candidate)
        for pair in relevant_pairs:
            other = next(iter(pair - {name}))
            if not self._has_assignment(other):
                continue
            z_other = self.point_assignments[self.points[other].z]
            zb_other = self.point_assignments[self.points[other].zb]
            if (
                sp.simplify(simplified_z - z_other) == 0
                and sp.simplify(simplified_zb - zb_other) == 0
            ):
                raise GeometryError(
                    f"Distinct constraint would be violated by assigning '{name}' = '{other}'."
                )

    # ------------------------------------------------------------------
    # Circle registration helpers
    # ------------------------------------------------------------------
    def register_circle(self, name: str, circle: Circle) -> Circle:
        """Register a circle under the provided name."""
        if name in self.circles:
            raise GeometryError(f"Circle '{name}' is already defined.")
        self.ensure_point(circle.center)
        self.circles[name] = circle
        return circle

    def circle_from_three_points(self, name: str, A: str, B: str, C: str) -> Circle:
        """Construct and register the circle passing through A, B, C."""
        for point in (A, B, C):
            self.ensure_point(point)
        center_label = self._unique_internal_name(f"circle_center_{name}")
        self.circumcenter(A, B, C, center_label)
        radius_sq = self.squared_distance(A, center_label)
        circle = Circle(
            center=center_label,
            radius_squared=sp.simplify(radius_sq),
            context=(A, B, C),
        )
        return self.register_circle(name, circle)

    def get_circle(self, name: str) -> Circle:
        """Return a previously registered circle."""
        try:
            return self.circles[name]
        except KeyError as exc:
            raise GeometryError(f"Circle '{name}' is not defined.") from exc

    def add_distinct_points(self, P: str, Q: str) -> None:
        """Declare that P and Q must represent distinct geometric points."""
        if P == Q:
            raise GeometryError("Distinct constraint requires two different labels.")
        self.ensure_point(P)
        self.ensure_point(Q)
        pair = frozenset((P, Q))
        if pair in self.distinct_pairs:
            return
        self.distinct_pairs.add(pair)
        self._validate_distinct_pair(P, Q)

    def _solve_point_from_equations(
        self,
        name: str,
        equations: Sequence[sp.Expr],
        *,
        avoid: Optional[Sequence[str]] = None,
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Solve a system for point `name` using the provided equations."""
        self.add_point(name)
        z_var, zb_var = self.z_symbol(name), self.zb_symbol(name)
        processed = [self._apply_all(eq) for eq in equations]
        solutions = sp.solve(processed, (z_var, zb_var), dict=True)
        if not solutions:
            raise GeometryError(
                f"Construction for point '{name}' failed: system has no solution."
            )
        chosen = None
        avoid_pairs: List[Tuple[sp.Expr, sp.Expr]] = []
        if avoid:
            for label in avoid:
                self.ensure_point(label)
                avoid_pairs.append((self.z(label), self.zb(label)))
        for candidate in solutions:
            z_candidate = sp.simplify(candidate[z_var])
            zb_candidate = sp.simplify(candidate[zb_var])
            if avoid_pairs:
                skip = False
                for z_avoid, zb_avoid in avoid_pairs:
                    if (
                        sp.simplify(z_candidate - z_avoid) == 0
                        and sp.simplify(zb_candidate - zb_avoid) == 0
                    ):
                        skip = True
                        break
                if skip:
                    continue
            chosen = candidate
            break
        if chosen is None:
            chosen = solutions[0]
        self._set_point_assignment(name, chosen[z_var], chosen[zb_var])
        return self.z(name), self.zb(name)

    def _tangent_intersection(
        self,
        P: str,
        Q: str,
        center: str,
        name: str,
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Solve for the intersection of tangents at P and Q to the circumcircle centered at `center`.
        """
        self.add_point(name)
        eq1 = self.perpendicular_poly(P, name, P, center)
        eq2 = self.perpendicular_poly(Q, name, Q, center)
        return self._solve_point_from_equations(name, [eq1, eq2])

    def _require_main_unit_triangle(self) -> UnitTriangleConfig:
        """Return the configured main unit triangle or raise a helpful error."""
        if self._main_unit_triangle is None:
            raise GeometryError(
                "Main unit triangle is not configured. Use set_main_unit_triangle first."
            )
        return self._main_unit_triangle

    def _unit_root_label(self, vertex: str) -> str:
        """Return the auxiliary root label for a vertex of the main unit triangle."""
        config = self._require_main_unit_triangle()
        normalized = vertex.upper()
        if normalized not in config.roots:
            raise GeometryError(
                f"Point '{vertex}' is not part of the main unit triangle."
            )
        return config.roots[normalized]

    # ------------------------------------------------------------------
    # Constraint inspection helpers
    # ------------------------------------------------------------------
    def _conjugate_free_expr(self, expr: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
        """
        Reduce a constraint polynomial and return the simplified numerator/denominator.

        The result is simplified as much as possible; residual conjugate symbols may
        remain if substitutions cannot eliminate them completely.
        """
        substituted = self._apply_all(expr)
        together_expr = sp.together(substituted)
        numerator, denominator = sp.fraction(together_expr)
        numerator = sp.expand(numerator)
        denominator = sp.expand(denominator)

        return numerator, denominator

    def _constraint_polynomials(
        self,
        constraint: str,
        args: Sequence[str],
        *,
        angle: Optional[sp.Expr] = None,
        directed: Optional[bool] = None,
    ) -> List[sp.Expr]:
        """
        Resolve the requested constraint into the underlying polynomial(s).
        """
        normalized = constraint.lower()
        if normalized in {"collinear", "line"}:
            if len(args) != 3:
                raise GeometryError(
                    "Collinear constraint expects exactly three point labels."
                )
            return [self.collinear_poly(*args)]
        if normalized in {"perpendicular", "perp"}:
            if len(args) != 4:
                raise GeometryError(
                    "Perpendicular constraint expects exactly four point labels."
                )
            return [self.perpendicular_poly(*args)]
        if normalized in {"concyclic", "cyclic"}:
            if len(args) != 4:
                raise GeometryError(
                    "Concyclic constraint expects exactly four point labels."
                )
            return [self.concyclic_poly(*args)]
        if normalized in {"angle", "angle_value"}:
            if len(args) != 3:
                raise GeometryError(
                    "Angle constraint expects three point labels (A, B, C)."
                )
            if angle is None:
                raise GeometryError("Angle constraint requires an 'angle' expression.")
            return [self.angle_value_poly(args[0], args[1], args[2], angle, raw=False)]
        if normalized in {"angle_bisector_either", "angle_bisector_any"}:
            if len(args) != 4:
                raise GeometryError(
                    "Angle bisector (either) constraint expects four point labels (A, B, C, D)."
                )
            return [self.angle_bisector_either_poly(*args)]
        if normalized == "circumcenter":
            if len(args) != 4:
                raise GeometryError(
                    "Circumcenter constraint expects four point labels (A, B, C, U)."
                )
            eq1, eq2 = self.circumcenter_polys(*args)
            return [eq1, eq2]
        if normalized in {"midpoint"}:
            if len(args) != 3:
                raise GeometryError(
                    "Midpoint constraint expects three point labels (P, Q, M)."
                )
            eq_z, eq_zb = self.midpoint_polys(*args)
            return [eq_z, eq_zb]
        if normalized in {"centroid"}:
            if len(args) != 4:
                raise GeometryError(
                    "Centroid constraint expects four point labels (A, B, C, G)."
                )
            eq_z, eq_zb = self.centroid_polys(*args)
            return [eq_z, eq_zb]
        if normalized in {"point_reflection", "reflection_point", "reflect_point"}:
            if len(args) != 3:
                raise GeometryError(
                    "Point reflection expects three point labels (P, O, Q)."
                )
            eq_z, eq_zb = self.point_reflection_polys(*args)
            return [eq_z, eq_zb]
        if normalized in {"line_reflection", "reflection_line", "reflect_line"}:
            if len(args) != 4:
                raise GeometryError(
                    "Line reflection expects four point labels (P, A, B, Q)."
                )
            return list(self.line_reflection_polys(*args))
        if normalized in {"isogonal_reflection", "angle_bisector_reflection"}:
            if len(args) != 5:
                raise GeometryError(
                    "Isogonal reflection expects five point labels (A, B, C, D, E)."
                )
            return [self.isogonal_reflection_poly(*args)]
        if normalized in {
            "isogonal_conjugate",
            "isogonal_conj",
            "add_isogonal_conjugate",
        }:
            if len(args) != 5:
                raise GeometryError(
                    "Isogonal conjugate expects five point labels (A, B, C, P, Q)."
                )
            return list(self.isogonal_conjugate_polys(*args))
        if normalized in {
            "triangle_similarity",
            "triangle_similarity_directed",
            "similar_triangle",
            "similar_triangle_directed",
        }:
            if len(args) != 6:
                raise GeometryError(
                    "Triangle similarity expects six point labels (A, B, C, D, E, F)."
                )
            directed_flag = True if directed is None else bool(directed)
            return list(self.triangle_similarity_polys(*args, directed=directed_flag))
        if normalized in {
            "triangle_similarity_undirected",
            "triangle_similarity_reflected",
            "similar_triangle_undirected",
            "similar_triangle_reflected",
        }:
            if len(args) != 6:
                raise GeometryError(
                    "Triangle similarity expects six point labels (A, B, C, D, E, F)."
                )
            return list(self.triangle_similarity_polys(*args, directed=False))
        if normalized in {
            "triangle_congruence",
            "triangle_equal",
            "triangle_congruent",
            "triangles_equal",
        }:
            if len(args) != 6:
                raise GeometryError(
                    "Triangle congruence expects six point labels (A, B, C, D, E, F)."
                )
            directed_flag = True if directed is None else bool(directed)
            return list(self.triangle_congruence_polys(*args, directed=directed_flag))
        if normalized in {
            "triangle_congruence_undirected",
            "triangle_congruence_reflected",
            "triangle_equal_undirected",
            "triangle_congruent_undirected",
        }:
            if len(args) != 6:
                raise GeometryError(
                    "Triangle congruence expects six point labels (A, B, C, D, E, F)."
                )
            return list(self.triangle_congruence_polys(*args, directed=False))

        if normalized == "equal_angle":
            if len(args) != 6:
                raise GeometryError(
                    "Equal angle expects six point labels (A, B, C, D, E, F)."
                )
            return [self.equal_angle_poly(*args)]
        if normalized == "equal_length":
            if len(args) != 4:
                raise GeometryError(
                    "Equal length expects four point labels (A, B, C, D)."
                )
            return [self.equal_length_poly(*args)]
        if normalized == "tangent_circles":
            if len(args) != 6:
                raise GeometryError(
                    "Tangent circles expects four point labels (A, B, C, D,E,F)."
                )
            return [self.tangent_circles_poly(*args)]
        raise GeometryError(f"Unsupported constraint '{constraint}'.")

    def constraint_conjugate_free(
        self,
        constraint: str,
        args: Sequence[str],
        *,
        angle: Optional[sp.Expr] = None,
        directed: Optional[bool] = None,
    ) -> List[Tuple[sp.Expr, sp.Expr]]:
        """
        Return conjugate-free numerator/denominator pairs for the chosen constraint.
        """
        polys = self._constraint_polynomials(
            constraint, args, angle=angle, directed=directed
        )
        return [self._conjugate_free_expr(poly) for poly in polys]

    def perpendicular_conjugate_free(
        self, A: str, B: str, C: str, D: str
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Backwards-compatible helper for AB ⟂ CD checks.
        """
        return self.constraint_conjugate_free("perpendicular", [A, B, C, D])[0]

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def format_symbol(self, symbol: sp.Symbol, *, style: str = "latex") -> str:
        """
        Return a string representation of a stored symbol with point labels instead of z_/zb_.
        """
        if style == "latex":
            return self._latex_symbol_names.get(symbol, sp.latex(symbol))
        if style == "text":
            if symbol in self._text_symbol_replacements:
                return sp.sstr(self._text_symbol_replacements[symbol])
            return sp.sstr(symbol)
        raise ValueError(f"Unsupported display style '{style}'.")

    def format_expr(self, expr: sp.Expr, *, style: str = "latex") -> str:
        """
        Render an expression using point labels for coordinates.
        """
        simplified = sp.simplify(expr)
        if style == "latex":
            return sp.latex(simplified, symbol_names=self._latex_symbol_names)
        if style == "text":
            replaced = simplified.xreplace(self._text_symbol_replacements)
            return sp.sstr(replaced)
        raise ValueError(f"Unsupported display style '{style}'.")

    def display_learned_rules(self, *, style: str = "latex") -> Dict[str, str]:
        """
        Return learned conjugate substitutions with formatted symbols/expressions.
        """
        return {
            self.format_symbol(symbol, style=style): self.format_expr(expr, style=style)
            for symbol, expr in self.learned_subs.items()
        }

    # ------------------------------------------------------------------
    # Introspection utilities
    # ------------------------------------------------------------------
    def learned_rules(self) -> Dict[str, sp.Expr]:
        """Return learned conjugate substitutions as a plain dictionary."""
        return {
            str(symbol): sp.simplify(expr) for symbol, expr in self.learned_subs.items()
        }

    def point_summary(
        self, names: Optional[Sequence[str]] = None, *, style: str = "latex"
    ) -> Dict[str, str]:
        """Return simplified coordinate expressions for selected points."""
        if names is None:
            names = sorted(self.points.keys())
        summary: Dict[str, str] = {}
        for name in names:
            summary[name] = self.format_expr(self.z(name), style=style)
            summary[f"\overline{{{name}}}"] = self.format_expr(
                self.zb(name), style=style
            )
        return summary

    def constraint_strings(
        self, *, style: str = "text", subs: bool = False
    ) -> List[str]:
        """Return stored constraints in their recorded polynomial form."""
        return [
            self.format_expr(
                sp.factor(
                    sp.expand(
                        constraint.subs(self._substitution_map(), simultaneous=False)
                        if subs
                        else constraint
                    )
                ),
                style=style,
            )
            for constraint in self.constraints
        ]
        return []


__all__ = [
    "GeometryEngine",
    "GeometryError",
    "Line",
]
