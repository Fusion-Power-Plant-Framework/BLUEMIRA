Designers and Builders
----------------------

Designers and Builders are the main structure in ``bluemira`` for creating a Component.

The two main classes are:

* :py:class:`bluemira.base.designer.Designer`: the Designer base class
* :py:class:`bluemira.base.builder.Builder`: the Builder base class

A :py:class:`~bluemira.base.designer.Designer` carries out component design activities,
for example evaluating a geometry parameterisation,
or performing an optimisation.
It outputs a minimal representation of a component that can be used to generate geometry
(note that this minimal representation may itself be some, simple, geometry).

A :py:class:`~bluemira.base.builder.Builder` is responsible for generating and manipulating ``bluemira``
geometry objects (CAD), to create a :py:class:`~bluemira.base.components.Component`.
It will typically perform geometry operations like offsets or sweeps,
and is not intended to perform any complex calculations.

Designers
^^^^^^^^^

:py:class:`~bluemira.base.designer.Designer`\s solve the minimal design problem of a
:py:class:`~bluemira.base.components.Component` or any other object that requires calculation or optimisation.
The minimal design problem could result in, for instance, a geometric wire
or some relevant values that allow the :py:class:`~bluemira.base.builder.Builder` to build
the :py:class:`~bluemira.base.components.Component` CAD.
A Designer is optional as some components can be completely built with
user input and calculations from previous component design stages.
Optimisation problems should be run within a :py:class:`~bluemira.base.designer.Designer`.

A basic :py:class:`~bluemira.base.designer.Designer` only requires a :py:meth:`run` method and a
:py:attr:`param_cls` attribute to store the :py:class:`ParameterFrame` class reference.
Once initialised the :py:class:`~bluemira.base.designer.Designer` is run with its :py:meth:`execute` method.
The below is for illustrative purposes and is overkill,
in this instance you would just have a :py:class:`~bluemira.base.builder.Builder`.

.. code-block:: python

    from dataclasses import dataclass

    from bluemira.base.designer import Designer
    from bluemira.base.parameter_frame import Parameter, ParameterFrame

    @dataclass
    class DesignerPF(ParameterFrame):
        A: Parameter[float]


    class MyDesigner(Designer[float])

        param_cls = DesignerPF

        def run(self) -> float:
            return  self.params.A.value


To initialise a :py:class:`~bluemira.base.designer.Designer` you need any :py:class:`ParameterFrame` instance that is a
superset of :py:attr:`Designer.param_cls` and optionally a `build_config` dictionary which
contains configuration options for the :py:class:`~bluemira.base.designer.Designer`.
It is possible to execute a :py:class:`~bluemira.base.designer.Designer` in different ways depending on requirements or
software availability. If another method such as :py:meth:`mock` or :py:meth:`read` is defined
and `run_mode` is specified in the `build_config` the :py:meth:`execute` method will call the specified method.

.. code-block:: python

    class MyOtherDesigner(Designer[float]):

        param_cls = DesignerPF

        def run(self) -> float:
            return self.params.A.value

        def mock(self) -> float:
            return self.params.A.value ** 2

    params = {"A": {"value": 2, "unit": "dimensionless"}}
    build_config = {"run_mode": "mock"}
    designer = MyOtherDesigner(params, build_config)
    print(designer.execute())  # == 4

Builders
^^^^^^^^

The minimal design problem output if required along with user input to the :py:class:`~bluemira.base.builder.Builder` is all
the information needed to build the CAD for the :py:class:`~bluemira.base.components.Component`.
The :py:meth:`build` method of the :py:class:`~bluemira.base.builder.Builder` usually builds
the xz, xy and xyz views of a :py:class:`~bluemira.base.components.Component`, and combines them into a component tree.
Although what the build method does is up to the :ref:`Reactor Designer <how to use>`.

It is recommended to build only one xyz sector for a given component.
The resulting reactor build will be much faster and the
:py:meth:`~bluemira.base.reactor.Reactor.show_cad` and :py:meth:`~bluemira.base.reactor.Reactor.save_cad` methods
provide a `n_sector` argument which will copy and rotate each sector as needed for a given usecase.

Like a :py:class:`~bluemira.base.designer.Designer` a :py:class:`~bluemira.base.builder.Builder` requires a :py:attr:`param_cls` attribute
and is initialised with a :py:class:`ParameterFrame` instance that is a superset of :py:attr:`param_cls`
and optionally the :py:meth:`Designer.execute` output and a `build_config` dictionary.
The below is an example of a standard :py:class:`~bluemira.base.builder.Builder` structure and once initialised
the :py:meth:`build` method is called to create the :py:class:`~bluemira.base.components.Component`.

.. code-block:: python

    from dataclasses import dataclass

    from bluemira.base.builder import Builder
    from bluemira.base.parameter_frame import Parameter, ParameterFrame

    @dataclass
    class BuilderPF(ParameterFrame):
        R_0: Parameter[float]


    class MyBuilder(Builder):

        param_cls = BuilderPF

        def build(self) -> Component
            return self.component_tree(
                xz=[self.build_xz()],
                xy=[self.build_xy()],
                xyz=[self.build_xyz()],
            )

        def build_xz(self):
            """Build a 2D geometry PhysicalComponent"""

        def build_xy(self):
            """Build a 2D geometry PhysicalComponent"""

        def build_xyz(self):
            """Build a 3D geometry PhysicalComponent"""
