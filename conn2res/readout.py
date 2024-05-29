# -*- coding: utf-8 -*-
"""
Functionality to train readout module
"""
import warnings
import numpy as np
import pandas as pd

from sklearn import linear_model
# from sklearn.base import is_classifier, is_regressor
# from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
# from sklearn.multiclass import OneVsRestClassifier

from . import utils
from . import performance


class Readout:
    """
    _summary_
    """
    def __init__(self, estimator=None, y=None):
        """
        _summary_

        Parameters
        ----------
        estimator : _type_, optional
            _description_, by default None
        y : _type_, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        if estimator is not None and y is not None:
            raise ValueError("y must be None if estimator is provided")
        elif estimator is not None and y is None:
            self.model = estimator
        elif estimator is None and y is not None:
            self.model = y
        else:
            self.model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, arg):
        """
        _summary_

        Parameters
        ----------
        arg : _type_
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(arg, str):
            #TODO: add other sklearn modules such as SVM
            #TODO: be able to pass arguments to model
            # (instead of arg being a str it could be
            # a dictionary: arg={'model_name':dict_of_args})
            self._model = getattr(linear_model, arg)()
        elif 'sklearn' in str(arg.__class__):
            self._model = arg
        elif isinstance(arg, (list, np.ndarray)):
            self._model = select_model(arg)
        else:
            raise TypeError(
                "arg must be either a str specifying a sklearn linear model, "
                "an instance of a sklearn model, "
                "or a list or numpy.ndarray of target values (y)"
            )

    def train(self, X, y, sample_weight=None):
        """
        _summary_

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        sample_weight : _type_, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        # check X and y are arrays
        X, y = _check_xy_type(X, y)

        # check sample_weight is an array
        if isinstance(sample_weight, (list, tuple)):
            sample_weight = utils.concat(sample_weight)

        # check X and y dimensions
        X = _check_x_dims(X)
        y = _check_y_dims(y)

        if len(X) != len(y):
            raise ValueError(
                "Number of samples in X is different from number of samples in y"
            )

        # TODO: define sample_weight
        # train model
        self._model.fit(X=X, y=y, sample_weight=sample_weight)

    def test(self, X, y, sample_weight=None, metric=None, **kwargs):
        """
        _summary_

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        sample_weight : _type_, optional
            _description_, by default None
        metric : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # check X and y are arrays
        X, y = _check_xy_type(X, y)

        # check sample_weight is an array
        if isinstance(sample_weight, (list, tuple)):
            sample_weight = utils.concat(sample_weight)

        # check X and y dimensions
        X = _check_x_dims(X)
        y = _check_y_dims(y)

        if len(X) != len(y):
            raise ValueError(
                "Number of samples in X is different from number of samples in y"
            )

        # assign value to metric if None
        if metric is None:
            metric = 'score'

        # make metric a tuple to enable different metrics
        #  on the same model
        if isinstance(metric, str):
            metric = (metric,)

        # estimate scores
        scores = dict()
        for m in metric:
            if m == 'score':
                # use default score method of model
                scores[m] = self._model.score(
                    X, y, sample_weight=sample_weight)
            else:
                # get score function
                func = getattr(performance, m)

                # predict values
                y_pred = self._model.predict(X)
                # estimate score
                scores[m] = func(
                    y, y_pred, sample_weight=sample_weight, **kwargs)

        return scores

    def run_task(
        self, X, y, sample_weight=None, frac_train=0.7, metric=None,
        readout_modules=None, readout_nodes=None, **kwargs
    ):
        """
        _summary_

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        sample_weight : _type_, optional
            _description_, by default None
        frac_train : float, optional
            _description_, by default 0.7
        readout_modules : _type_, optional
            _description_, by default None
        readout_nodes : _type_, optional
            _description_, by default None
        metric : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        TypeError
            _description_
        ValueError
            _description_
        """
        # get train_test split for X and y
        try:
            (x_train, x_test), (y_train, y_test) = X, y
        except ValueError as exc:
            if not utils.check(X, y):
                xy_names = [type(X).__name__, type(y).__name__]

                raise TypeError(
                    f"X is {xy_names[0]} and y is {xy_names[1]}. X and y must be the same type"
                ) from exc

            x_train, x_test, y_train, y_test = train_test_split(
                    X, y, frac_train=frac_train)

        # define sample_weight
        if isinstance(sample_weight, (list, tuple)):
            sample_weight_train, sample_weight_test = sample_weight
        else:
            sample_weight_train, sample_weight_test = _get_sample_weight(
                (y_train, y_test), split_set=sample_weight
            )

        # define set(s) of readout nodes
        if readout_modules is not None and readout_nodes is not None:
            raise ValueError(
                "Only one of readout_nodes or readout_modules must be passed"
            )
        elif readout_modules is not None and readout_nodes is None:
            readout_nodes, ids = get_readout_nodes(readout_modules)


        # train and test model
        if readout_nodes is None:

            self.train(
                x_train, y_train, sample_weight_train
            )

            score = self.test(
                x_test, y_test, sample_weight_test, metric=metric, **kwargs
            )

            df_scores = pd.DataFrame(data=score, index=[0])

        elif isinstance(readout_nodes, (list, tuple, np.ndarray)):

            #TODO: allow per_trial test
            # if isinstance(x_train, (list, tuple)):
            #     sections = utils.get_sections(x_train)
            #     convert_to_list = True
            # else:
            #     convert_to_list = False

            # convert to arrays to enable indexing with readout_nodes
            x_train, y_train = _check_xy_type(x_train, y_train)
            x_test, y_test = _check_xy_type(x_test, y_test)

            # readout_nodes is an array of arrays
            if all(isinstance(i, (list, tuple, np.ndarray)) for i in readout_nodes):
                df_scores = []
                for i, readouts in zip(ids, readout_nodes):
                    self.train(
                        x_train[:, readouts], y_train,
                        sample_weight_train
                    )

                    score = self.test(
                        x_test[:, readouts], y_test,
                        sample_weight_test, metric=metric, **kwargs
                    )

                    df = pd.DataFrame(data=score, index=[0])
                    df['module'] = i
                    df['n_nodes'] = len(readouts)
                    df_scores.append(df[['module', 'n_nodes'] + metric])

                df_scores = pd.concat(df_scores)

            # readout_nodes is a single array
            else:
                self.train(
                    x_train[:, readout_nodes], y_train,
                    sample_weight_train
                )

                score = self.test(
                    x_test[:, readout_nodes], y_test,
                    sample_weight_test, metric=metric, **kwargs
                )

                df_scores = pd.DataFrame(data=score, index=[0])
        print("\ndf_scores --------\n")
        print(df_scores,"\n\n")
        return df_scores


def select_model(y):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    def isinteger(a):
        return np.equal(np.mod(a, 1), 0)

    # if list or tuple convert to array
    if isinstance(y, (list, tuple)):
        y = utils.concat(y)

    if y.dtype in [np.int32, np.int64]:
        if y.squeeze().ndim == 1:
            if len(np.unique(y)) == 2:
                return classifier()  # 'classification'
            else:
                return multiclass_classifier()  # 'multiclass_classification'
        elif y.squeeze().ndim == 2:
            return multioutput_classifier()  # 'multioutput_classification' + 'multioutput_multiclass_classification'
        else:
            raise ValueError("Target variable y has to be 1D or 2D")

    elif y.dtype in [np.float32, np.float64]:
        if y.squeeze().ndim == 1:
            # double check that values are actually continuos
            is_integer = isinteger(y)
            if all(is_integer):
                if len(np.unique(y)) == 2:
                    return classifier()  # 'classification'
                else:
                    return multiclass_classifier()  # 'multiclass_classification'
            else:
                return regressor()  # 'regression'
        elif y.squeeze().ndim == 2:
            # double check that values are actually continuos
            is_integer = [
                all(isinteger(y[:, col])) for col in range(y.shape[1])
            ]
            if all(is_integer):
                return multioutput_classifier()  # 'multioutput_classification'
            else:
                return multioutput_regressor()  # 'multioutput_regression'
        else:
            raise ValueError("Target variable y has to be 1D or 2D")


def regressor(*args, **kwargs):
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    return linear_model.Ridge(alpha=0.5, fit_intercept=False, *args, **kwargs)


def classifier(*args, **kwargs):
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    return linear_model.RidgeClassifier(alpha=0.0, fit_intercept=False, *args, **kwargs)


def multioutput_regressor(*args, **kwargs):
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    # TODO: return MultiOutputRegressor(regressor(*args, **kwargs))
    # MultiOutputRegressor does not handle decision_function for
    # plotting diagnostics curve
    return regressor(*args, **kwargs)


def multioutput_classifier(*args, **kwargs):
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    # TODO: return MultiOutputClassifier(classifier(*args, **kwargs))
    # MultiOutputClassifier does not handle decision_function for
    # plotting diagnostics curve
    return classifier(*args, **kwargs)


def multiclass_classifier(*args, **kwargs):
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    # TODO: return OneVsRestClassifier(classifier(*args, **kwargs))
    # OneVsRest does not handle sample_weight
    return classifier(*args, **kwargs)


def train_test_split(*args, frac_train=0.7, n_train=None):
    """
    Splits data into training and test sets according to
    'frac_train'

    Parameters
    ----------
    frac_train : float, from 0 to 1
        fraction of samples in training set
    n_train : int (optional)
        number of training samples

    Returns
    -------
    train-test splits : tuple
        tuple containing train-test split of inputs.
    """
    argout = []
    for arg in args:
        if n_train is None and isinstance(arg, list):
            n_train = int(frac_train * len(arg))
        if n_train is None and isinstance(arg, np.ndarray):
            n_train = int(frac_train * arg.shape[0])
        argout.extend([arg[:n_train], arg[n_train:]])

    return tuple(argout)


def _check_xy_type(X, y):
    """
    _summary_

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if X is not None and isinstance(X, (list, tuple)):
        X = utils.concat(X)

    if y is not None and isinstance(y, (list, tuple)):
        y = utils.concat(y)

    return X, y


def _check_x_dims(X):
    """
    Check that X have the right dimensions

    Parameters
    ----------
    X : numpy.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """

    if X.ndim == 1:
        return X[:, np.newaxis]
    else:
        return X


def _check_y_dims(y):
    """
    Check that y have the right dimensions

    Parameters
    ----------
    y : numpy.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """

    return y.squeeze()


def _get_sample_weight(y, split_set=None):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_
    split_set : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    try:
        y_train, y_test = y
    except ValueError:
        y_train = y
        y_test  = y

    sample_weight_train, sample_weight_test = None, None

    if split_set == 'train':
        sample_weight_train = _sample_weight(y_train, split_set)

    elif split_set == 'test':
        sample_weight_test = _sample_weight(y_test, split_set)

    elif split_set == 'both':
        sample_weight_train = _sample_weight(y_train, split_set)
        sample_weight_test = _sample_weight(y_test, split_set)

    return sample_weight_train, sample_weight_test


def _sample_weight(y, split_set, seed=None):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_
    split_set : _type_
        _description_
    seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
        seed to initialize the random number generator, by default None
        for details, see numpy.random.default_rng()

    Returns
    -------
    _type_
        _description_
    """

    # get baseline value and type. If y is multi-
    # target take only last baseline value.
    baseline = _baseline(y)[-1]
    baseline_type = _baseline_class(y)

    # convert y to array
    if isinstance(y, (list, tuple)):
        sections = utils.get_sections(y)
        y = utils.concat(y)
        convert_to_list = True
    else:
        convert_to_list = False

    # if y is multi-target take only last target
    if y.ndim == 2:
        y = y[:, -1]

    # create sample_weight
    sample_weight = np.ones_like(y).astype(float)
    if baseline_type == 'class1':
        sample_weight[y == baseline] = 0

    elif baseline_type == 'class2':
        sample_weight[y == baseline] = 0

        # split sample weight in trials
        sample_weight = utils.split(sample_weight, sections)

        # estimate average length of label across trials
        lens = int(np.mean(
            [len(np.where(i == 1)[0]) for i in sample_weight.copy() if len(set(i)) > 1]
            ))

        # add weights to trials where label = baseline
        for i in range(len(sample_weight)):
            if all(sample_weight[i] == 0):
                sample_weight[i][-lens:] = 1

        sample_weight = utils.concat(sample_weight)

    elif baseline_type == 'class3':
        pass

    if split_set == 'train':
        print('-----------------------------------')

        # use random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        idx = np.where(sample_weight == 0)[0]
        sample_weight[idx] = rng.rand((len(idx)))

    if convert_to_list:
        sample_weight = utils.split(sample_weight, sections)

    return sample_weight


def _baseline(y):
    """
    Return baseline value for each target in y

    Parameters
    ----------
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    if isinstance(y, (list, tuple)):
        y = utils.concat(y)

    y = _check_y_dims(y)

    if y.ndim == 1:
        n_targets = 1
        values, counts = np.unique(y, return_counts=True)
        baseline = values[counts == counts.max()]
    elif y.ndim == 2:
        n_targets = y.shape[1]
        baseline = []
        for target in range(n_targets):
            values, counts = np.unique(y[:, target], return_counts=True)
            baseline.extend(values[counts == counts.max()])

    if not len(baseline) == n_targets:
        warnings.warn("There is more than one baseline value per target")

    return baseline


def _baseline_class(y):
    """
    Define the type of baseline based on two boolean flags:
    (flag1) 'baseline_exists'. True if labels are different
    across trials
    (flag2) 'baseline_included'. True if baseline value is
    also a label.

    If baseline_exists is True and baseline_included = False,
    then baseline_class = 1
    If baseline_exists is True and baseline_included = True,
    then baseline_class = 2
    If baseline_exists is False then baseline_class = 3

    Parameters
    ----------
    y : _type_
        _description_
    baseline : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    # get baseline value. If y is multi-
    # target take only last baseline value
    baseline = _baseline(y)[-1]

    # get labels per trial
    labels_per_trial = []
    for trial in y:
        # if y is multitarget take only last target
        if trial.ndim == 2:
            trial = trial[:, -1]

        values, counts = np.unique(trial, return_counts=True)
        diff_from_baseline = np.setdiff1d(values, baseline)

        if diff_from_baseline.size == 0:
            labels_per_trial.append(baseline)
        elif diff_from_baseline.size == 1:
            labels_per_trial.append(diff_from_baseline[0])
        elif diff_from_baseline.size > 1:
            labels_per_trial.append(diff_from_baseline[-1])

    # flag1 : baseline_exists
    baseline_exists = True
    if len(np.unique(labels_per_trial)) == 1:
        baseline_exists = False

    # flag2 : baseline_included (as a label)
    baseline_included = False
    if baseline in labels_per_trial:
        baseline_included = True

    # determine baseline_type
    if baseline_exists and not baseline_included:
        baseline_type = 'class1'
    elif baseline_exists and baseline_included:
        baseline_type = 'class2'
    elif not baseline_exists:
        baseline_type = 'class3'

    return baseline_type


def get_readout_nodes(readout_modules):
    """
    Return a list with the set(s) of nodes in each module in
    'readout_modules', plus a set of module ids

    Parameters
    ----------
    readout_modules : (N,) list, tuple, numpy.ndarray or dict
        Can be a 1D array-like that assigns modules to each node. Can
        be a list of lists, where each sublist corresponds to the
        indexes of subsets of nodes. Can be a dictionary key:val pairs,
        where the keys correspond to modules and the values correspond
        to list/tuple that contains the subset of nodes in each module.

    Returns
    -------
    readout_nodes : list
        list that contains lists with indexes of subsets of nodes in
        'readout_modules'
    ids : list
        list that contains lists with indexes of subsets of nodes in
        'readout_modules'

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(readout_modules, (list, tuple, np.ndarray)):
        if all(isinstance(i, (list, tuple, np.ndarray)) for i in readout_modules):
            ids = list(range(len(readout_modules)))
            readout_nodes = list(module for module in readout_modules)
        else:
            ids = list(set(readout_modules))
            readout_nodes = list(
                np.where(np.array(readout_modules) == i)[0] for i in ids
            )
    elif isinstance(readout_modules, dict):
        ids = list(readout_modules.keys())
        readout_nodes = list(readout_modules.values())
    else:
        raise TypeError("")

    return readout_nodes, ids


def _get_sample_weight_old(inputs, labels=None, sample_block=None):
    """
    Time averages dataset based on sample class and sample weight

    Parameters
    ----------
    inputs : numpy.ndarray or list of numpy.ndarrays
        input data
    labels: numpy.ndarray or list of numpy.ndarrays
        label data
    sample_block : numpy.ndarray
        block structure which is used as a basis for weighting
        (i.e., same weights are applied within each block)

    Returns
    -------
    sample_weight: numpy.ndarray or list of numpy.ndarrays
        weights of samples which can be used either for averaging time
        series or training models whilst weighting samples in the cost
        function
    idx_sample: numpy.ndarray or list of numpy.ndarrays
        indexes of samples with one index per block (see sample_block)
    """
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]

    # if isinstance(labels, np.ndarray):
    #     labels = [labels]

    sample_weight = []
    if sample_block is None:
        for data in inputs:
            # sample block based on unique combinations of classes in data
            icol = [col for col in range(data.shape[1]) if np.unique(
                data[:, col]).size <= 3]  # class is based on <=3 real values

            _, sample_block = np.unique(
                data[:, icol], return_inverse=True, axis=0)

            # get unique sample blocks
            _, ia, nc = np.unique(
                sample_block, return_index=True, return_counts=True)

            # sample weight
            sample_weight.append(
                np.hstack([np.tile(1/e, e) for e in nc[np.argsort(ia)]]))

    else:
        # get unique sample blocks
        _, ia, nc = np.unique(
            sample_block, return_index=True, return_counts=True)

        for data in inputs:
            # sample weight
            sample_weight.append(
                np.hstack([np.tile(1/e, e) for e in nc[np.argsort(ia)]]))


    return sample_weight