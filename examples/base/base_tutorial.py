# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
A little tutorial on how to make use some of the framework functionality in the
BLUEPRINT framework.

You are meant to step your way through each line of this script, and
introspect the objects you are creating along the way.
Feel free to change parameters!
"""

# %%
from typing import Type
from BLUEPRINT.base import ParameterFrame, ReactorSystem, TypeFrameworkError
from BLUEPRINT.geometry.loop import Loop


# %%[markdown]
# # Anatomy of a ReactorSystem

# %%
class TypicalSystem(ReactorSystem):
    """
    The class docstring. It should contain a simple description of what
    the class is.
    """

    # This is where we define class attributes (we can give type-hints)

    name: str
    config: Type[ParameterFrame]
    inputs: dict

    something: Type[Loop]

    # Here we define the Parameters that the TypicalSystem class will use
    # These will be pulled from the Reactor that we are making when using this
    # class

    # fmt: off  --> this is the flag for black to ignore this part of the code
    default_params = [
        ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
        ["coolant", "Coolant", "Water", None, "Divertor coolant type", "Common sense"],
        ["T_in", "Coolant inlet T", 80, "°C", "Coolant inlet T", None],
    ]
    # fmt: on   --> turn formatting back on!

    def __init__(self, config: Type[ParameterFrame], inputs: dict) -> None:
        self.config = config
        self.inputs = inputs

        # Here are going to update the default Parameters with the config
        # (which normally comes from the Reactor)
        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

    def add_loop(self, loop: Type[Loop]) -> None:
        """
        Assign a loop to the class.

        Parameters
        ----------
        loop: Loop
            The Loop to use
        """
        self.something = loop


# %%[markdown]
# Now let's make an instance of the TypicalSystem
#
# First, we make a ParameterFrame, which we will make really small here, but in
# practice will come from a Reactor, and have lots of things in it

# %%
config = ParameterFrame(
    [
        ["n_TF", "Number of TF coils", 18, "N/A", None, "Input"],
        ["n_PF", "Number of PF coils", 8, "N/A", None, "Input"],
    ]
)
inputs = {"something": "random"}


typ_system = TypicalSystem(config, inputs)

# %%[markdown]
# Notice 3 things:
# *  the class default for n_TF was 16, and we passed a config with 18.
#    Now the TypicalSystem thinks n_TF = 18
# *  The config ParameterFrame had a Parameter n_PF, which wasn't defined
#    in the TypicalSystem class. TypicalSystem ignores this Parameter: it
#    simply doesn't update its ParameterFrame
# *  Some Parameters were in TypicalSystem, that were not updated by our
#    "global" config. They are left untouched.

# %%
print(typ_system.params)

# %%[markdown]
# OK now let's add a Loop to our TypicalSystem

# %%
loop = Loop(x=[1, 2, 3, 4, 5, 1], y=[0, 1, 2, 3, 0, 0])

typ_system.add_loop(loop)

# %%[markdown]
# So far, so good. But what about all these type annotations?
# Python is dynamically typed, and sometimes this can cause problems, in
# particular when doing a lot of very different things...
# Using some magic in the base module, our type annotations are now strictly
# enforced by default.

# So let's try and break things...

# %%
try:
    typ_system.name = 42
except TypeFrameworkError as e:
    print(e)

# %%
try:
    typ_system.add_loop(34.6)
except TypeFrameworkError as e:
    print(e)

# %%
try:
    typ_system.something = "nothing"
except TypeFrameworkError as e:
    print(e)
