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
Machine Learning which is a fancy name for data dimensionality reduction
"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline


class ModelFitting:
    def __init__(
        db: pd.DataFrame, targets: List[str], target: str, *pipeline_components
    ):
        self._db = db
        self.set_targets(targets, target)
        self.train_test(split=None)  # Default is 0.8
        self.setup_pipeline(pipeline_components)

    @property
    def db(self):
        return self._db

    def set_targets(self, targets: List[str], target: str):
        """
        Remove the result columns from the DataFrame, and set the fitting target.

        Parameters
        ----------
        targets: List[str]
            The list of column names in the DataFrame which are result columns.
            These are excluded from the fitting.
        target: str
            The result column name in the DataFrame to fit to
        """
        inputs = self.db.copy()
        for targ in targets:
            inputs = inputs.drop(targ, axis=1)
        self.inputs = inputs
        self.target = self.db[target]
        self.variables = inputs.columns

    def setup_pipeline(pipeline):
        self.pipeline = make_pipeline(*pipeline)

    def train_test(self, split=None):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        split: Union[float, None]
            The split fraction between train and test. Defaults to 0.8
        """
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            idx_train,
            idx_test,
        ) = train_test_split(
            self.inputs, self.target, np.arange(len(self.inputs)), test_size=split
        )
        self.c_test = self.constant[idx_test]
        self.c_train = self.constant[idx_train]


def linear(data, constant=0):
    """
    Linear to linear transform (returns original data)
    """
    return data + constant


def power_forward(data):
    """
    Power Law to Linear transform
    """
    return np.log(data)


def power_reverse(data, constant=0):
    """
    Linear to Power law transform
    """
    return np.exp(data) + constant


class LinearRegression:
    def __init__(self, forward_func, reverse_func):
        self.forward_func = forward_func
        self.reverse_func = reverse_func

    def linearisation(self, data):
        """
        Apply function to linearise data
        """
        return self.forward_func(data)

    def delinearise(self, data):
        """
        Apply function to delinearise dat
        """
        return self.reverse_func(data)


class PCA:
    def __init__(self, kernel=False, **kwargs):

        if kernel:
            if not kwargs:
                kwargs = dict(
                    n_components=None, kernel="rbf", fit_inverse_transform=True
                )
            self.pca = KernelPCA(**kwargs)
        else:
            if not kwargs:
                kwargs = dict(n_components="mle")
            self.pca = PCA(**kwargs)

    def fit(self):

        self.pca.fit(data)

    def fittransform(self):

        self.pca.fit_transform(data)

    def prediction():

        pca.components_, pca.explained_variance_
        X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)
