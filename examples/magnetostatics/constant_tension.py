import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import MU_0
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)


class DummyToroidalFieldSolver:

    def field(x, y, z):
        return np.array([np.zeros_like(x), 1.0/x, np.zeros_like(z)])
    
def calculate_discrete_constant_tension_shape(
    r1: float,
    r2: float,
    n_tf: int,
    tf_wp_width: float,
    tf_wp_depth: float,
    n_points: int,
    solver=ArbitraryPlanarRectangularXSCircuit,
    tolerance: float =1e-3,
    plot: bool =False,
    include_inboard: bool =False,
):
    """
    Calculate a "constant tension" shape for a TF coil winding pack, for a discrete number
    of TF coils.

    Parameters
    ----------
    r1:
        Inboard TF winding pack centreline radius
    r2:
        Outboard TF winding pack centreline radius
    n_tf:
        Number of TF coils
    tf_wp_width:
        Radial extent of the TF coil WP
    tf_wp_depth:
        Toroidal extent of the TF coil WP
    n_points:
        Number of points in the TF coil (no guarantees on output size)
    solver:
        Solver object which provides the magnetic field calculation
    tolerance:
        Tolerance for convergence [m]
    plot:
        Whether or not to plot

    Notes
    -----
    This procedure numerically calculates a constant tension shape for a TF coil
    by assuming a circle first and using magnetostatic solvers to integrate the 
    toroidal field along the shape. Iteration is used to modify the shape until
    convergence.

    Note that the tension is constant only along the centreline. This is an
    approximation, that I hope should keep the average tension in a coil of non-zero thickness
    relatively low, but no promises.


    The current is not required, but would be technically for the absolute value of the
    tension.

    A rectangular cross-section is assumed.

    The procedure was originally developed by Dr. L. Giannini for use with ANSYS, and
    has been quite heavily modified here.

    BiotSavartFilament is a poor choice of solver for this procedure, but does yield
    interesting results.

    Using the associated DummyToroidalFieldSolver, one can pretty perfectly recreate
    the closed-form solution for the Princeton-D.
    """
    n_points //= 2  # We solve for a half-coil
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    sin_theta = np.sin(theta)
    r = (r2 + r1) / 2 + (r2 - r1) / 2 * np.cos(theta + np.pi / 2)
    z = (r2 - r1) / 2 * np.sin(theta + np.pi / 2)
    r = r[:n_points]
    z = z[:n_points]
    ra = r.copy()
    za = z.copy()

    errorr = 1.0
    errorz = 1.0
    iter_count = 0

    while (errorz > tolerance or errorr > tolerance) and iter_count < 100:
        iter_count += 1
        rs = np.r_[r[::-1], r[1:]]
        zs = np.r_[z[::-1], -z[1:]]

        if plot:
            plt.figure(1)
            plt.plot(rs, zs, ".")
            plt.axis("equal")
            plt.ylabel("z [m]", fontsize=20)
            plt.xlabel("r [m]", fontsize=20)
            plt.draw()
            plt.pause(0.001)
        
        if solver == BiotSavartFilament and include_inboard:
            # Improve B-S discretisation at the inboard
            dl_0 = np.hypot(rs[1]-rs[0], zs[1]-zs[0])
            dl_straight = np.hypot(rs[-1]-rs[0], zs[-1]-zs[0])
            n_inboard = dl_straight / dl_0
            n_inboard = int(max(3, np.ceil(n_inboard)))
            
            r_straight = np.linspace(rs[-1], rs[0], n_inboard)[1:-1]
            z_straight = np.linspace(zs[-1], zs[0], n_inboard)[1:-1]
            rc = np.concatenate([rs,  r_straight])
            zc = np.concatenate([zs,  z_straight])
        else:
            rc = rs
            zc = zs
     
        coordinates = Coordinates({"x": rc, "y": 0.0, "z": zc})
        coordinates.close()
        coordinates.set_ccw([0, -1, 0])

        if solver == ArbitraryPlanarRectangularXSCircuit:
            filament = ArbitraryPlanarRectangularXSCircuit(
                coordinates, 0.5 * tf_wp_width, 0.5 * tf_wp_depth, 1.0
            )
            cage = HelmholtzCage(filament, n_tf)
        elif solver == BiotSavartFilament:
            radius = (0.5 * tf_wp_width + 0.5 * tf_wp_depth) * 0.5
            filament = BiotSavartFilament(coordinates, radius=radius, current=1.0)
            cage = HelmholtzCage(filament, n_tf)
        elif solver == DummyToroidalFieldSolver:
            cage = DummyToroidalFieldSolver
        else:
            raise ValueError(f"Not a valid solver: {solver}")

        B = cage.field(rs[:n_points], np.zeros_like(rs)[:n_points], zs[:n_points])
        Btor = B[1, :]
        rr_intb =  r[::-1]
        rr = r[::-1]      

        Btor = 2 * np.pi * Btor / (MU_0 * n_tf)

        intB = np.zeros(n_points)

        for i in range(1, n_points):
            intB[i] = intB[i - 1] + 0.5 * (Btor[i - 1] + Btor[i]) * (rr_intb[i] - rr_intb[i - 1])

        intB_fh = interp1d(rr_intb, intB, kind="linear", fill_value="extrapolate")
        interpolator = interp1d(rr, Btor, kind="linear", fill_value="extrapolate")

        T = MU_0 * n_tf / (8 * np.pi) * intB[-1]
        k = 4 * np.pi * T / (MU_0 * n_tf)

        x0 = r2
        Btor = Btor[::-1]
        for i in range(n_points):
            xx = x0
            error = 1.0
            inner_iter = 0
            while error > tolerance:
                inner_iter += 1
                F = intB_fh(x0) + k * (sin_theta[i] - 1)
                xx = x0 - F / interpolator(x0)
                error = abs(xx - x0) / abs(x0)
                x0 = xx
                if inner_iter > 50:
                    print("WARNING: inner iterations = 50")
                    break
            r[i] = xx
            if i != 0:
                z[i] = z[i - 1] - k * (
                    sin_theta[i - 1] / Btor[i - 1] + sin_theta[i] / Btor[i]
                ) / 2 * (theta[i] - theta[i - 1])

        errorr = np.linalg.norm(r - ra) / np.linalg.norm(r)
        errorz = np.linalg.norm(z - za) / np.linalg.norm(z)

        ra = r.copy()
        za = z.copy()
        print(
            f"Iteration: {iter_count} | T = {T * 1e-6} | r error = {errorr} | z error = {errorz}"
        )

    r = np.concatenate((r, [r[-1]]))
    z = np.concatenate((z, [0]))
    r = r[::-1]
    z = z[::-1]

    if plot:
        # Plot the normalised tension
        theta = np.concatenate((theta, [np.pi / 2])) * 180 / np.pi
        theta = theta[::-1]
        rho = np.zeros(n_points + 1)
        Tc = np.zeros(n_points + 1)
        dzdr = np.zeros(n_points + 1)
        dzdr2 = np.zeros(n_points + 1)
        Btor = Btor[::-1]

        for i in range(2, n_points):
            h = r[i + 1] - r[i]
            theta = r[i] - r[i - 1]
            dzdr[i] = (z[i + 1] - z[i - 1]) / (h + theta)
            dzdr2[i] = (
                2
                * (z[i + 1] + z[i - 1] - 2 * z[i] - dzdr[i] * (h - theta))
                / (h**2 + theta**2)
            )
            rho[i] = (1 + dzdr[i] ** 2) ** (3 / 2) / dzdr2[i]
            Tc[i] = (
                -rho[i]
                / 2
                * (Btor[i] * n_tf * MU_0 / (2 * np.pi))
            )

        plt.figure(100)
        plt.plot(r[2:-1], Tc[2:-1] * 1e-6, "r", linewidth=2)
        plt.show()

    return r, z


if __name__ == "__main__":

    import os

    from bluemira.base.file import get_bluemira_path
    from bluemira.geometry.coordinates import Coordinates

    path = get_bluemira_path("magnetostatics", subfolder="examples")

    r1 = 1.18
    r2 = 4.7198
    n_tf =  12
    current = 1.0
    tf_wp_width = 0.2
    tf_wp_depth = 0.4
    n_points = 100

    rPD, zPD = PrincetonD._princeton_d(r1, r2, 0.0, 200)
    rPD = rPD[100:]
    zPD = zPD[100:]

    ra, za = calculate_discrete_constant_tension_shape(
        r1,
        r2,
        n_tf,
        tf_wp_width,
        tf_wp_depth,
        n_points,
        solver=DummyToroidalFieldSolver,
        tolerance=1e-3,
        plot=False,
    )
    rap, zap = calculate_discrete_constant_tension_shape(
        r1,
        r2,
        n_tf,
        tf_wp_width,
        tf_wp_depth,
        n_points,
        solver=ArbitraryPlanarRectangularXSCircuit,
        tolerance=1e-3,
        plot=False,
    )

    f, ax = plt.subplots()
    ax.plot(rPD, zPD, color="r", label="Princeton-D")
    ax.plot(ra, za, label=f"Numerical 1/r", ls="--", color="g")
    ax.plot(rap, zap, label=f"Numerical true integral", color="b")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend()
    plt.show()
