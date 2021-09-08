# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
An example file to make the Tapered Picture Frame version of the ToroidalFieldCoils
object, optimized for the minimum length

"""

import os
import matplotlib.pyplot as plt
from BLUEPRINT.base.file import make_BP_path
from BLUEPRINT.base import ParameterFrame
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from BLUEPRINT.equilibria.shapes import flux_surface_manickam
from BLUEPRINT.cad.model import CADModel

# BASED ON GV_SCR_03 from the PROCESS-STEP repository
# fmt: off
params = [
    ["R_0", "Major radius", 3.639, "m", None, "Input"],
    ["B_0", "Toroidal field at R_0", 2.0, "T", None, "Input"],
    ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
    ["tk_tf_nose", "TF coil inboard nose thickness", 0.0377, "m", None, "Input"],
    ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.02, 'm', None, 'Input'],
    ["tk_tf_wp", "TF coil winding pack thickness", 0.569, "m", None, "PROCESS"],
    ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02, "m", None, "Input"],
    ["tk_tf_ins", "TF coil ground insulation thickness", 0.008, "m", None, "Input"],
    ["tk_tf_insgap", "TF coil WP insertion gap", 1.0E-7, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
    ["r_tf_in", "Inboard radius of the TF coil inboard leg", 0.148, "m", None, "PROCESS"],
    ["TF_ripple_limit", "Ripple limit constraint", 0.65, "%", None, "Input"],
    ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
    ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
    ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.8934, "m", None, "PROCESS"],
    ["h_cp_top", "Height of the Tapered Section", 6.199, "m", None, "PROCESS"],
    ["r_cp_top", "Radial Position of Top of taper", 0.8934, "m", None, "PROCESS"],
    ["tf_wp_depth", "TF coil winding pack depth (in y)", 0.4625, "m", "Including insulation", "PROCESS"],
    ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
    ['h_tf_max_in', 'Plasma side TF coil maximum height', 12.0, 'm', None, 'PROCESS'],
    ["r_tf_curve", "Radial position of the CP-leg conductor joint", 1.5, "m", None, "PROCESS"],
    ['tk_tf_outboard', 'TF coil outboard thickness', 0.569, 'm', None, 'Input', 'PROCESS'],

]
# fmt: on

parameters = ParameterFrame(params)

read_path = make_BP_path("Geometry", subfolder="data/BLUEPRINT")
write_path = make_BP_path("CP_Coil", subfolder="generated_data/BLUEPRINT")

lcfs = flux_surface_manickam(3.42, 0, 2.137, 2.9, 0.55, n=40)
lcfs.close()

# lcfs.translate([0.15, 0, 0])

name = os.sep.join([read_path, "KOZ_PF_test1.json"])
ko_zone = Loop.from_file(name)

to_tf = {
    "name": "Example_PolySpline_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "CP",  # This is the overall coil shape parameterisation to use
    "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
    "npoints": 800,
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 3,  # This is the number of current filaments to use in y
    "nr": 2,  # This is the number of current filaments to use in x
    "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,  # This is the path that the shape will be read from
    "write_folder": write_path,  # This is the path that the shape will be written to
}

tf1 = ToroidalFieldCoils(parameters, to_tf)

tf1.optimise()

f1, ax1 = plt.subplots()
ko_zone.plot(ax1, edgecolor="b", fill=False)
tf1.plot_ripple(ax=ax1)
plt.gca().set_aspect("equal")
# plt.show()

f, ax = plt.subplots()
tf1.plot_xy(ax=ax)
plt.show()

n_tf = tf1.params.n_TF
model = CADModel(n_tf)
model.add_part(tf1.build_CAD())
model.display(pattern="tq")
model.save_as_STEP_assembly(write_path)
