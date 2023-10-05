# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import numpy as np
import pytest

from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter
from bluemira.builders.coil_supports import (
    ITERGravitySupportBuilder,
    ITERGravitySupportBuilderParams,
    OISBuilder,
    OISBuilderParams,
    PFCoilSupportBuilder,
    PFCoilSupportBuilderParams,
    StraightOISDesigner,
    StraightOISDesignerParams,
    StraightOISOptimisationProblem,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD, TripleArc
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    circular_pattern,
    make_circle_arc_3P,
    make_polygon,
    mirror_shape,
    offset_wire,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire


class TestITERGravitySupportBuilder:
    pd = PrincetonD()
    pd.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    pd.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    pd_xz_koz = pd.create_shape()
    pf = PictureFrame()
    pf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    pf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    pf_xz_koz = pf.create_shape()
    ta = TripleArc()
    ta.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    ta.adjust_variable("f2", value=2, lower_bound=2, upper_bound=4)
    ta_xz_koz = ta.create_shape()

    tf_kozs = (pd_xz_koz, pf_xz_koz, ta_xz_koz)

    @staticmethod
    def _make_builder(tf, **kwargs):
        defaults = {
            "x_g_support": {"value": 10, "unit": "m"},
            "z_gs": {"value": -20, "unit": "m"},
            "tf_wp_depth": {"value": 1.4, "unit": "m"},
            "tf_wp_width": {"value": 0.8, "unit": "m"},
            "tk_tf_side": {"value": 0.05, "unit": "m"},
            "tf_gs_tk_plate": {"value": 0.025, "unit": "m"},
            "tf_gs_g_plate": {"value": 0.025, "unit": "m"},
            "tf_gs_base_depth": {"value": 2.4, "unit": "m"},
        }
        for k, v in kwargs.items():
            defaults[k]["value"] = v

        params = ITERGravitySupportBuilderParams.from_dict(defaults)
        return ITERGravitySupportBuilder(params, {}, tf)

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("x_gs", [0, 2, 3.74, 14.56, 100])
    def test_bad_support_radius(self, tf, x_gs):
        builder = self._make_builder(tf, x_g_support=x_gs)
        with pytest.raises(BuilderError):
            builder.build()

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("z_gs", [0, -2, 100])
    def test_bad_support_height(self, tf, z_gs):
        builder = self._make_builder(tf, z_gs=z_gs)
        with pytest.raises(BuilderError):
            builder.build()

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("x_gs", [3.75, 7, 10])
    def test_good_support_radius(self, tf, x_gs):
        builder = self._make_builder(tf, x_g_support=x_gs)
        component = builder.build()

        assert len(component.get_component("xyz").children) == 1
        assert len(component.get_component("xz").children) == 1
        assert len(component.get_component("xy").children) == 0


class TestPFCoilSupportBuilder:
    my_test_params = PFCoilSupportBuilderParams(
        Parameter("tf_wp_width", 0.8, unit="m"),
        Parameter("tf_wp_depth", 1.4, unit="m"),
        Parameter("tk_tf_side", 0.05, unit="m"),
        Parameter("pf_s_tk_plate", 0.15, unit="m"),
        Parameter("pf_s_n_plate", 4, unit=""),
        Parameter("pf_s_g", 0.05, unit="m"),
    )

    tf = PrincetonD()
    tf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    tf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    tf.adjust_variable("dz", value=0, lower_bound=-10, upper_bound=24)
    tf_xz_koz = tf.create_shape()

    @staticmethod
    def make_dummy_pf(xc, zc, dxc, dzc):
        """
        Flake8
        """
        my_dummy_pf = PictureFrame()
        my_dummy_pf.adjust_variable("ri", value=0.1, lower_bound=0, upper_bound=np.inf)
        my_dummy_pf.adjust_variable("ro", value=0.1, lower_bound=0, upper_bound=np.inf)
        my_dummy_pf.adjust_variable(
            "x1", value=xc - dxc, lower_bound=0, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "x2", value=xc + dxc, lower_bound=0, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "z1", value=zc + dzc, lower_bound=-np.inf, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "z2", value=zc - dzc, lower_bound=-np.inf, upper_bound=np.inf
        )
        return my_dummy_pf.create_shape()

    @pytest.mark.parametrize(
        ("x", "z", "dx", "dz"),
        [
            (2.5, 10, 0.4, 0.5),
            (6, 12.5, 0.5, 0.5),
            (10, 11, 0.5, 0.67),
            (10, 11, 0.5, 1.25),
            (14.5, 6, 0.2, 0.5),
        ],
    )
    def test_good_positions_up_down_equal_volume(self, x, z, dx, dz):
        pf_xz_upper = self.make_dummy_pf(x, z, dx, dz)
        pf_xz_lower = self.make_dummy_pf(x, -z, dx, dz)

        upper_builder = PFCoilSupportBuilder(
            self.my_test_params, {}, self.tf_xz_koz, pf_xz_upper
        )
        upper_support = (
            upper_builder.build().get_component("xyz").get_component(upper_builder.name)
        )

        lower_builder = PFCoilSupportBuilder(
            self.my_test_params, {}, self.tf_xz_koz, pf_xz_lower
        )
        lower_support = (
            lower_builder.build().get_component("xyz").get_component(upper_builder.name)
        )
        np.testing.assert_almost_equal(
            lower_support.shape.volume, upper_support.shape.volume
        )

    @pytest.mark.parametrize(
        ("x", "z", "dx", "dz"),
        [
            (100, 0, 0.3, 0.3),
            (0.3, 0.3, 0.3, 0.3),
        ],
    )
    def test_bad_positions(self, x, z, dx, dz):
        pf_xz = self.make_dummy_pf(x, z, dx, dz)
        builder = PFCoilSupportBuilder(self.my_test_params, {}, self.tf_xz_koz, pf_xz)
        with pytest.raises(BuilderError):
            builder.build()


class TestOISBuilder:
    x_1 = 4
    x_2 = 16
    pd = PrincetonD({"x1": {"value": x_1}, "x2": {"value": x_2}}).create_shape()

    n_TF = 16
    hd = 0.5 * 1.6
    xs = make_polygon(
        {"x": [x_1 - hd, x_1 + hd, x_1 + hd, x_1 - hd], "y": [-hd, -hd, hd, hd], "z": 0},
        closed=True,
    )

    tf_coil = sweep_shape(xs, pd)

    def _check_no_intersection_with_TFs(self, ois, builder, tf_coils):
        ois_body = ois.get_component("xyz").get_component(f"{builder.RIGHT_OIS} 1").shape
        result = sorted(boolean_cut(ois_body, tf_coils[0]), key=lambda s: -s.volume)
        assert np.isclose(ois_body.volume, result[0].volume)
        ois_body = ois.get_component("xyz").get_component(f"{builder.LEFT_OIS} 1").shape
        result = sorted(boolean_cut(ois_body, tf_coils[0]), key=lambda s: -s.volume)
        assert np.isclose(ois_body.volume, result[0].volume)

    def _check_no_intersection_when_patterned(self, ois, builder, n_TF):
        tf_angle = 2 * np.pi / n_TF
        direction = (-np.sin(0.5 * tf_angle), np.cos(0.5 * tf_angle), 0)
        right_ois_0 = (
            ois.get_component("xyz").get_component(f"{builder.RIGHT_OIS} 1").shape
        )
        left_ois_1 = mirror_shape(right_ois_0, base=(0, 0, 0), direction=direction)
        full_ois = boolean_fuse([right_ois_0, left_ois_1])
        assert np.isclose(full_ois.volume, 2 * right_ois_0.volume)

    @pytest.mark.parametrize("n_TF", [14, 15, 16, 17, 18, 19])
    def test_rectangular_profile(self, n_TF):
        ois_profile = make_polygon(
            {"x": [self.x_2, self.x_2 + 0.5, 14.5, 14], "y": 0, "z": [0, 0, 6, 6]},
            closed=True,
        )
        params = OISBuilderParams(
            Parameter("n_TF", n_TF),
            Parameter("tf_wp_depth", 1.4),
            Parameter("tk_tf_side", 0.1),
        )
        builder = OISBuilder(params, {}, ois_profile)
        ois = builder.build()
        tf_coils = circular_pattern(self.tf_coil, n_shapes=n_TF)[:2]
        self._check_no_intersection_with_TFs(ois, builder, tf_coils)
        self._check_no_intersection_when_patterned(ois, builder, n_TF)

    @pytest.mark.parametrize("n_TF", [14, 15, 16, 17, 18, 19])
    def test_curved_profile(self, n_TF):
        result = boolean_cut(
            self.pd,
            make_polygon({"x": [9, 20, 20, 9], "z": [-4, -4, -8, -8]}, closed=True),
        )[-1]
        result.translate(vector=(-0.25, 0, 0))
        r_copy = result.deepcopy()
        r_copy.translate(vector=(0.5, 0, 0))
        p1 = result.start_point()
        p2 = result.end_point()
        p3 = r_copy.start_point()
        p4 = r_copy.end_point()

        join_1 = make_polygon([p2, p4])
        join_2 = make_polygon([p3, p1])

        ois_profile = BluemiraWire([result, join_1, r_copy, join_2])
        params = OISBuilderParams(
            Parameter("n_TF", n_TF),
            Parameter("tf_wp_depth", 1.4),
            Parameter("tk_tf_side", 0.1),
        )
        builder = OISBuilder(params, {}, ois_profile)
        ois = builder.build()
        tf_coils = circular_pattern(self.tf_coil, n_shapes=n_TF)[:2]
        self._check_no_intersection_with_TFs(ois, builder, tf_coils)
        self._check_no_intersection_when_patterned(ois, builder, n_TF)


class TestStraightOISDesigner:
    tf_wp_depth = 1.4
    tk_tf_side = 0.1
    n_TF = 16
    y_width = tf_wp_depth + 2 * tk_tf_side
    pd = PrincetonD().create_shape()
    pd2 = offset_wire(pd, 1.0)
    tf_xz_face = BluemiraFace([pd2, pd])
    keep_out_zones = (
        BluemiraFace(
            make_polygon({"x": [6, 12, 12, 6], "z": [0, 0, 15, 15]}, closed=True)
        ),
        BluemiraFace(
            make_polygon({"x": [0, 20, 20, 0], "z": [-1, -1, 1, 1]}, closed=True)
        ),
    )
    keep_out_zones2 = (
        BluemiraFace(
            make_polygon({"x": [4, 12, 12, 4], "z": [0, 0, 15, 15]}, closed=True)
        ),
        BluemiraFace(
            make_polygon({"x": [0, 20, 20, 0], "z": [-1, -1, 1, 1]}, closed=True)
        ),
    )

    params = StraightOISDesignerParams(
        tk_ois=Parameter("tk_ois", 0.3),
        g_ois_tf_edge=Parameter("g_ois_tf_edge", 0.2),
        min_OIS_length=Parameter("min_OIS_length", 1),
    )

    @pytest.mark.parametrize(
        ("koz", "n_ois"), [(keep_out_zones, 3), (keep_out_zones2, 2)]
    )
    def test_that_the_right_number_of_OIS_are_made(self, koz, n_ois):
        designer = StraightOISDesigner(self.params, {}, self.tf_xz_face, koz)
        ois_wires = designer.run()
        assert len(ois_wires) == n_ois

    def test_that_gradient_based_optimiser_works(self):
        wire = make_polygon({"x": [9, 8, 7, 6, 5, 4], "z": [0, 1, 2, 3, 3.5, 4]})
        wire = make_circle_arc_3P([9, 0, 0], [7, 0, 2], [4, 0, 4])
        keep_out_zone = BluemiraFace(
            make_polygon({"x": [6, 7, 7, 6], "z": [0, 0, 1.8, 1.8]}, closed=True)
        )

        opt_problem = StraightOISOptimisationProblem(wire, keep_out_zone)
        result_1 = opt_problem.optimise(
            x0=np.array([0.0, 1.0]),
            algorithm="COBYLA",
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 1000},
        ).x

        opt_problem = StraightOISOptimisationProblem(wire, keep_out_zone)
        result_2 = opt_problem.optimise(
            x0=np.array([0.0, 1.0]),
            algorithm="SLSQP",
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 1000},
        ).x

        length_1 = result_1[1] - result_1[0]
        length_2 = result_2[1] - result_2[0]
        # Alright so SLSQP isn't going to do as well as COBYLA on this one, but at
        # least the gradients aren't too wrong.
        assert np.isclose(length_1, length_2, rtol=0.01)
