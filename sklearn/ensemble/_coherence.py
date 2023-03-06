"""Coherence Classification

This module contains a coherence classification method.

The module contains:

    - The `BalsubramaniFreundClassifier` implements coherence classification
    with an arbitrary ensemble of classifiers.
"""

# Author: Steven L. An <sla001@eng.ucsd.edu>
#
# License: BSD 3-Clause

import warnings
from numbers import Integral, Real

import operator
import numpy as np
import cvxpy as cp

from scipy.sparse import coo_array, vstack
from statsmodels.stats.proportion import proportion_confint

from ._base import _BaseHeterogeneousEnsemble
from ._base import _fit_single_estimator
from ..base import ClassifierMixin
from ..base import clone

from ..metrics import accuracy_score, confusion_matrix

from ..preprocessing import LabelEncoder, OneHotEncoder
from ..utils import Bunch
from ..utils.parallel import delayed, Parallel
from ..utils.validation import check_is_fitted
from ..utils.validation import check_X_y
from ..utils.validation import _num_samples
from ..utils._param_validation import Interval, StrOptions
#from sklearn.utils.validation import check_array
#from sklearn.utils.multiclass import unique_labels

# pylint: disable=too-many-instance-attributes
class BalsubramaniFreundClassifier(_BaseHeterogeneousEnsemble, ClassifierMixin):
    """ The Balsubramani-Freund classifier.

    Parameters
    ----------
    estimators : list of (name, scikit-learn classifiers) pairs
        A list of classifiers that form the ensemble.
    unknown_ensemble : {True, False}, default=False
        True when you only have the ensemble predictions and the relevant
        estimator parameters (accuracies/class accuracies/confusion matrix
        entries and class frequencies), but you do NOT have the classifiers
        in the ensemble themselves.  Will need to use `fit_unknown_ensemble`
        to set the estimator parameters rather than the usual `fit`.
    loss : {'0-1', 'Xent'}, default='Xent'
        The loss function to be used with Balsubramani-Freund, 0-1 or cross
        entropy (aka maximum entropy problem).
    constraint : {'accuracy', 'class_accuracy', 'confusion_matrix'},\
                 default='accuracy'
        Determines whether the convex program constraints will be 'accuracy'
        (1 constraint per classifier), 'class_accuracy' (k constraints per
        classifier), or 'confusion_matrix' (k^2 constraints per classifier).
    use_equality_constraints : {True, False}, default=False
        Whether to use equality constraints in the convex program, i.e. do we
        know the exact accuracy and the class distribution of the dataset we
        want to predict on.
    prog_type : {'orig', 'joint_compact'},\
                 default='joint_compact'
        ``orig`` givesthe convex program presented by Balsubramani & Freund,
        which is good when you may increase classifiers but not datapoints,
        ``joint_compact`` gives the convex program where duplicate points,
        i.e. ones with the same ensemble predictions are removed, while
    prefit : {True, False}, default=False
        Whether or not the `estimators` provided have already been fitted.  If
        set to True, the estimators will not be refitted.
    lb_type : {'wilson', 'agresti_coull', 'beta', 'binom_test', None}\
            , default='wilson'
        What confidence interval if any to use to lower bound estimated
        accuries coming from ``fit``.  If None, then no lower bounding is done
        on the estimated accuracies/class accuracies/confusion matrix entries
        and class frequencies.
    signif_lvl : float, default=0.05
        Significance level for the confidence interval, i.e. 0.05 will result
        in a 95% confidence interval.  Must be between (0, 1).
    pred_type: {'determ', 'prob'}, default='determ'
        What type of predictions each estimator in the ensemble makes,
        deterministic or probabilistic.
    use_parameters : {True, False}, default=False
        Whether or not to make the constants in the convex program (accuracies,
        class accuracies, confusion matrix entries, class frequencies)
        parameters in the CVXPY sense, allowing for fast re-solves.
    solver : str, default=None
        Specify which solver for CVXPY to use, if not the default one that
        CVXPY chooses.
    verbose : {True, False}, default=False
        Whether or not the CXVPY output should be verbose, good for debugging.
    n_job : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib`.parallel_backend` context.
        ``-1`` means using all processors.  See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    _parameter_constraints: dict = {
        'estimators': [list, None],
        'unknown_ensemble': ['boolean'],
        'loss': [StrOptions({'0-1', 'Xent'})],
        'constraint': [StrOptions({'accuracy', 'class_accuracy', \
                'confusion_matrix'})],
        'use_equality_constraints':['boolean'],
        'prog_type': [StrOptions({'orig', 'joint_compact'})],
        'prefit': ['boolean'],
        'lb_type': [StrOptions({'wilson', 'agresti_coull','beta',\
                'binom_est'}), None],
        'signif_lvl': [Interval(Real, 0, 1, closed='neither')],
        'pred_type': [StrOptions({'determ', 'prob'})],
        'use_parameters': ['boolean'],
        'solver': [StrOptions(set(cp.installed_solvers())), None],
        'verbose': ['boolean'],
        'n_jobs': [None, Integral],
            }

    #pylint: disable=too-many-arguments
    def __init__(
        self,
        estimators,
        unknown_ensemble=False,
        loss='Xent',
        constraint='accuracy',
        use_equality_constraints=False,
        prog_type='joint_compact',
        prefit=False,
        lb_type='wilson',
        signif_lvl=0.05,
        pred_type='determ',
        use_parameters=False,
        solver=None,
        verbose=False,
        n_jobs=None,
        ):

        super().__init__(estimators=estimators)
        self.estimators=estimators
        self.unknown_ensemble = unknown_ensemble
        self.loss = loss
        self.constraint = constraint
        self.use_equality_constraints = use_equality_constraints
        self.prog_type = prog_type
        self.prefit = prefit
        self.lb_type = lb_type
        self.signif_lvl = signif_lvl
        self.pred_type = pred_type
        self.use_parameters = use_parameters
        self.solver = solver
        self.verbose = verbose
        self.n_jobs = n_jobs

    #pylint: disable=attribute-defined-outside-init
    def _validate_estimators(self):
        names, all_estimators = super()._validate_estimators()

        # if pred_type='prob', we require estimators to predict probabilities.
        if self.pred_type == 'prob':
            for i, (name, est) in enumerate(zip(names, all_estimators)):
                if not hasattr(est, "predict_proba"):
                    raise TypeError(
                        "BalsubramaniFreundClassifer with pred_type='prob'"
                        "requires that the weak learner support the"
                        "calculation of class probabilities with a"
                        "predict_proba method.\n"
                        f"Please change the estimator with {name} and "
                        f"index {i} or set pred_type to 'determ'."
                            )

        return names, all_estimators

    def _check_has_predicted_joint(self):
        """
        Check to see if a prediction has been made with a program with type
        prog_type='joint_compact'
        """

        return hasattr(self, "cp_var_joint_preds_")

    def _validate_params(self):
        super()._validate_params()

        if self.unknown_ensemble:
            warnings.warn(
                    "unknown_ensemble has been set to true.  Arguments for the"
                    "following parameters will be ignored: estimators, prefit,"
                    " lb_type, signif_level, n_job.  Make sure to use "
                    "fit_unknown_ensemble instead of fit."
                    )

        #change the normalization factor
        if not self.unknown_ensemble and self.estimators is None:
            raise ValueError("Must have non- None value for `estimators` "
                    "parameter when `unknown_ensemble`=False.")

        if self.pred_type == 'prob' and self.prog_type == 'joint_compact':
            raise ValueError(f"Program type {self.prog_type} must be used with\
                    deterministic predictions, i.e. pred_type='determ'")
        return self

    def _get_prediction_patterns(self, predictions, pattern_cts):
        """
        Get the prediction patterns of the ensemble.
        """
        # compute the number of unique patterns
        self.predictions_ = np.array(predictions)
        self.unique_cols_, self.col_uniq_inds_, self.col_inv_, self.col_cts_ = \
                np.unique(self.predictions_, axis=1, return_counts=True,\
                return_inverse=True, return_index=True)

        if pattern_cts is not None:
            # recompute the stuff from above to expand the data
            if pattern_cts.ndim != 1:
                raise ValueError("pattern_cts must have 1 dimension"
                        f"but has {pattern_cts.ndim}.")
            self.col_cts_ = pattern_cts
            self.col_uniq_inds_ = np.zeros(len(pattern_cts))
            self.col_inv_ = np.zeros(np.sum(pattern_cts))

            offset = 0
            for i in range(len(pattern_cts) - 1):
                new_ind = pattern_cts[i] + offset
                self.col_uniq_inds_[i + 1] = new_ind
                self.col_inv_[offset:new_ind] = np.array([i] * pattern_cts[i])
                offset += pattern_cts[i]


        self.n_uniq_cols_ = self.unique_cols_.shape[1]


    def _validate_cvx_program_params(self):
        def _check_if_valid_probs(distri, name):
            # check all values between 0 and 1 inclusive
            try:
                if not all(0 <= ele <= 1 for ele in distri):
                    raise ValueError(f"{name} must be between 0 and 1.")
            except TypeError:
                if not 0 <= distri <= 1:
                    raise ValueError(f"{name} must be between 0 and 1.")

            if np.sum(distri) > 1:
                raise ValueError(f"{name} must not sum to over 1.")

        _check_if_valid_probs(self.class_freq_lb_, 'Class frequencies')

        for i in range(self.n_estimators_):
            if self.constraint == 'accuracy':
                _check_if_valid_probs(self.accuracy_lb_[i],\
                        f"Estimator {i}'s accuracy")
            else:
                for j in range(self.n_classes_):
                    if self.constraint == 'class_accuracy':
                        _check_if_valid_probs(self.class_accuracy_lb_[i][j],\
                                f"Estimator {i}'s class accuracies")
                    elif self.constraint == 'confusion_matrix':
                        _check_if_valid_probs(self.confusion_matrix_lb_[i][j,],\
                                f"Estimator {i}'s confusion matrix row {j}")

    def _confint(self, count, nobs):
        return proportion_confint(count, nobs, alpha=self.signif_lvl,\
                method=self.lb_type)

    #pylint: disable=too-many-branches
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.unknown_ensemble:
            raise ValueError("Cannot use fit when unknown_ensemble is True.")
        check_X_y(X, y)
        self._validate_params()
        names, all_estimators = self._validate_estimators()

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        # Put fitted estimators in a list
        if self.prefit:
            self.estimators_ = []
            for estimator in all_estimators:
                if estimator != 'drop':
                    check_is_fitted(estimator)
                    self.estimators_.append(estimator)
        else:
            # Otherwise fit them, then put in list
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_single_estimator)(clone(est), X,\
                            transformed_y)
                    for est in all_estimators
                    if est != 'drop'
                    )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != 'drop':
                current_est = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_est
                est_fitted_idx += 1
            else:
                self.named_estimators_[name_est] = "drop"

        self.predict_method_ = 'predict' if self.pred_type=='determ'\
                else 'predict_proba'

        # Generate predictions from now fit estimators.
        predictions = [
                getattr(estimator, self.predict_method_)(X)
                for estimator in all_estimators
                if estimator != 'drop'
                ]

        # compute the number of estimators in the ensemble
        self.n_estimators_ = len(predictions)

        # calculate class frequencies
        self.n_classes_ = len(self.classes_)
        self.class_freq_cts_ = np.bincount(self.le_.transform(y))

        # calculate accuracies/class accuracies/confusion matrices
        if self.constraint == 'accuracy':
            self.accuracy_counts_ = [accuracy_score(transformed_y, pred,\
                    normalize=False) for pred in predictions]
        elif self.constraint is ('class_accuracy' or 'confusion_matrix'):
            self.confusion_matrix_counts_ = [confusion_matrix(transformed_y,\
                    pred) for pred in predictions]

            if self.constraint == 'class_accuracy':
                self.class_accuracy_counts = [np.diag(cm_counts)\
                        for cm_counts in self.confusion_matrix_counts_]

        # compute the counts needed for confidence interval/conversion from
        # counts to probabilities.
        num_samples = _num_samples(X)

        # perform lower bounding if asked for
        if self.lb_type is None:
            self.class_freq_lb_ = self.class_freq_cts_ / y.size
            if self.constraint == 'accuracy':
                self.accuracy_lb_ = self.accuracy_counts_ / num_samples
            elif self.constraint == 'class_accuracy':
                self.class_accuracy_lb_ = [
                        class_acc_count / num_samples\
                        for class_acc_count in self.class_accuracy_counts]
            elif self.constraint == 'confusion_matrix':
                self.confusion_matrix_lb_ = [
                        conf_mat_count / num_samples\
                        for conf_mat_count in self.confusion_matrix_counts_]
        else:
            self.class_freq_lb_, _ = \
                    self._confint(self.class_freq_cts_, \
                    [y.size] * self.n_classes_)
            if self.constraint == 'accuracy':
                self.accuracy_lb_ = \
                        [self._confint(acc_count, num_samples)\
                        for acc_count in self.accuracy_counts_]
            elif self.constraint == 'class_accuracy':
                self.class_accuracy_lb_, _ = \
                        [self._confint(class_acc_count, num_samples)\
                        for class_acc_count in self.class_accuracy_counts]
            elif self.constraint == 'confusion_matrix':
                self.confusion_matrix_lb_, _ = \
                        [self._confint(conf_mat_count, num_samples)\
                        for conf_mat_count in self.confusion_matrix_counts_]

        if self.constraint is ('class_accuracy' or 'confusion_matrix'):
            self.estimator_pred_class_freqs_ = [self.class_freq_lb_
                    for i in range(self.n_estimators_)]

        return self

    def fit_unknown_ensemble(
            self,
            estimator_params,
            class_freqs,
            classes,
            n_estimators,
            ):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        self.le_ = LabelEncoder().fit(classes)
        self.classes_ = self.le_.classes_

        self.n_classes_ = len(self.classes_)
        self.n_estimators_ = n_estimators

        # do consistency checks
        if self.n_classes_ != len(class_freqs):
            raise ValueError(f'{self.n_classes_} classes, yet there are '
                    f'{len(class_freqs)} many class frequencies.')
        if self.n_estimators_ != len(estimator_params):
            raise ValueError(f'{self.n_estimators_} estimators, yet there are '
                    f'{len(estimator_params)} many sets of estimator '
                    'parameters.')

        # no need to check when using accuracies since the check is the same
        # as above
        if self.constraint == 'class_accuracy':
            for i in range(self.n_estimators_):
                if len(estimator_params[i]) != self.n_classes_:
                    raise ValueError(f"Estimator {i} has "
                            f"{len(estimator_params[i])} class accuracies, "
                            f"rather than the expected {self.n_classes_}")
        elif self.constraint == 'confusion_matrix':
            for i in range(self.n_estimators_):
                if estimator_params[i].shape !=\
                        (self.n_classes_, self.n_classes_):
                    raise ValueError(f"Estimator {i} has confusion matrix of "
                            f"size {estimator_params[i].shape}, "
                            f"rather than the expected ({self.n_classes_}, "
                            f"{self.n_classes_}).")

        self.class_freq_lb_ = class_freqs

        if self.constraint == 'accuracy':
            self.accuracy_lb_ = estimator_params
        elif self.constraint == 'class_accuracy':
            self.class_accuracy_lb_ = estimator_params
        elif self.constraint == 'confusion_matrix':
            self.confusion_matrix_lb_ = estimator_params

        self._validate_cvx_program_params()

        return self

    def _make_orig_cvx_prog_constraints(self, predictions):
        """ Makes the Balsubrani-Freund convex program a la the original paper.
        """
        # set the comparison we want to use
        if self.use_equality_constraints:
            oper = operator.eq
        else:
            oper = operator.ge

        constraints = []

        if self.use_parameters:
            #assign parameters to rhs
            self.cp_rhs_class_freq_ = cp.Parameter(self.n_classes_, nonneg=True)
            self.cp_rhs_class_freq_.value = self.class_freq_lb_

            if self.constraint == 'accuracy':
                self.cp_rhs_accs_ = cp.Parameter(self.n_estimators_, nonneg=True)
                self.cp_rhs_accs_.value = self.accuracy_lb_
            elif self.constraint == 'class_accuracy':
                self.cp_rhs_class_accuracies_ = cp.Parameter(self.n_estimators_ * self.n_classes_, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_class_accuracies_[i * self.n_classes_ : (i + 1) * self.n_classes_].value = self.class_accuracy_lb_[i]
            else:
                # number of entries in a confusion matrix
                cm_len = np.power(self.n_classes_, 2)
                self.cp_rhs_confusion_matrix_ = cp.Parameter(self.n_estimators_ *\
                        cm_len, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_confusion_matrix_[i * cm_len : (i + 1) * cm_len] = np.ravel(self.confusion_matrix_lb_[i])

        else:
            #set RHS to usual suspects
            self.cp_rhs_class_freq_ = self.class_freq_lb_

            if self.constraint == 'accuracy':
                self.cp_rhs_accs_ = self.accuracy_lb_
            elif self.constraint == 'class_accuracy':
                self.cp_rhs_class_accuracies_ = self.class_accuracy_lb_
            else:
                self.cp_rhs_confusion_matrix_ = self.confusion_matrix_lb_

        # convert predictions into 1 hot encoding and make the resulting
        # matrix sparse if deterministic predictions.
        if self.pred_type == 'determ':
            #for each classifier, convert all their predictions to 1 hot
            # then append their predictions
            self.oh_le_ = OneHotEncoder(sparse_output=True)
            self.oh_le_.fit(np.arange(self.n_classes_).reshape(-1, 1))
            predictions = [
                    self.oh_le_.transform(prediction.reshape(-1,1)).toarray()
                    for prediction in predictions]

        variables = cp.Variable((self.n_samples_in_pred_x_, self.n_classes_))

        for i in range(self.n_estimators_):
            prediction_mass = np.sum(predictions[i])
            if self.constraint == 'accuracy':
                constraints.append(oper(
                    cp.sum(
                        cp.multiply(predictions[i], variables)),
                    self.cp_rhs_accs_[i] * prediction_mass))

            elif self.constraint == 'class_accuracy':
                constraints.append(oper(
                    cp.sum(cp.multiply(predictions[i], variables)
                        , axis=0)
                    , self.cp_rhs_class_accuracies_[i] * prediction_mass))
            else:
                # Convert the confusion matrix into a matrix dealing with
                # joint probabilities, Pr(Y, h_i(X)) since when confusion
                # matrix entries are estimated via labeled data, we are
                # estimating Pr(Y | h_i(X)).  This is done by multiplying the
                # lower bounded class frequencies, one class per row.

                cm_len = np.power(self.n_classes_, 2)
                lhs_ravel = cp.reshape(variables.T\
                        @ predictions[i], (cm_len, 1), order='C')
                rhs_ravel = cp.reshape(self.cp_rhs_confusion_matrix_[i] *\
                        prediction_mass, (cm_len, 1), order='C')
                constraints.append(oper(lhs_ravel, rhs_ravel))

        # make class frequency constraints
        constraints.append(oper(
            cp.sum(variables, axis=0),
            self.class_freq_lb_ * self.n_samples_in_pred_x_))

        # enforce each prediction being a probability distribution
        constraints += [cp.sum(variables, axis=1) == 1,
                variables >= 0]

        return variables, constraints

    def _make_constraint_rhs(self):
        """ Makes the RHS to the C matrix from the new paper
        """

        if self.use_parameters:
            #assign parameters to rhs
            self.cp_rhs_class_freq_ = cp.Parameter(self.n_classes_, nonneg=True)
            self.cp_rhs_class_freq_.value = self.class_freq_lb_

            if self.constraint == 'accuracy':
                self.cp_rhs_accs_ = cp.Parameter(self.n_estimators_, nonneg=True)
                self.cp_rhs_accs_.value = self.accuracy_lb_

            elif self.constraint == 'class_accuracy':
                self.cp_rhs_joint_class_accs_ = cp.Parameter(self.n_estimators_ * self.n_classes_, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_joint_class_accs_[i * self.n_classes_ : (i + 1) * self.n_classes_].value = self.class_accuracy_lb_[i]

            else:
                # number of entries in a confusion matrix
                cm_len = np.pow(self.n_classes_, 2)
                self.cp_rhs_joint_conf_mat_ = cp.Parameter(self.n_estimators_ *\
                        cm_len, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_joint_conf_mat_[i*cm_len : (i + 1)*cm_len].value\
                            = np.ravel(self.confusion_matrix_lb_[i])

        else:
            #set RHS to usual suspects
            self.cp_rhs_class_freq_ = self.class_freq_lb_

            if self.constraint == 'accuracy':
                self.cp_rhs_accs_ = self.accuracy_lb_
            elif self.constraint == 'class_accuracy':
                self.cp_rhs_joint_class_accs_ = np.zeros(self.n_estimators_ * self.n_classes_)
                for i in range(self.n_estimators_):
                    self.cp_rhs_joint_class_accs_[i * self.n_classes_ : (i + 1) * self.n_classes_] = self.class_accuracy_lb_[i]
            else:
                cm_len = np.power(self.n_classes_, 2)
                self.cp_rhs_joint_conf_mat_ = np.zeros(self.n_estimators_ *\
                        cm_len)

                for i in range(self.n_estimators_):
                    self.cp_rhs_joint_conf_mat_[i*cm_len : (i + 1)*cm_len]\
                            = np.ravel(self.confusion_matrix_lb_[i])

    def _make_constraint_lhs_and_rhs(self, predictions, pattern_cts):
        """ Makes the C matrix from the new paper
        """

        # get patterns
        self._get_prediction_patterns(predictions, pattern_cts)

        # make C matrix now
        c_blocks = []
        for i in range(self.n_estimators_):
            if self.constraint == 'accuracy':
                preds = self.unique_cols_[i, :].reshape(-1, 1)
                col = []

                for j in range(self.n_uniq_cols_):
                    col.append(j * self.n_classes_ + preds[j,0])

                col = np.array(col)
                row = np.array([0] * len(col))
                data = np.array([1] * len(col))
                C = coo_array((data, (row, col)), shape=(1, self.n_uniq_cols_ * self.n_classes_))
                #c_blocks.append(preds_one_hot)
                c_blocks.append(C)

            elif self.constraint == 'class_accuracy':
                # construct a sparse matrix for every classifier
                row = []
                col = []

                for j in range(self.n_uniq_cols_):
                    row.append(self.unique_cols_[i, j])
                    col.append(j * self.n_classes_ + self.unique_cols_[i, j])

                row = np.array(row)
                col = np.array(col)
                data = [1] * len(row)
                C = coo_array((data, (row, col)),
                        shape=(self.n_classes_, self.n_uniq_cols_ * self.n_classes_))
                c_blocks.append(C)

            else:
                # l ranges through the true classes
                for l in range(self.n_classes_):
                    # construct a sparse matrix for every classifier
                    row = []
                    col = []

                    for j in range(self.n_uniq_cols_):
                        row.append(self.unique_cols_[i, j])
                        col.append(j * self.n_classes_ + l)

                    row = np.array(row)
                    col = np.array(col)
                    data = [1] * len(row)
                    C = coo_array((data, (row, col)), shape=\
                            (self.n_classes_, self.n_uniq_cols_ * self.n_classes_))
                    c_blocks.append(C)

        self.c_ = vstack(c_blocks, 'coo')
        self._make_constraint_rhs()

    def _make_joint_cvx_prog_constraints(self, predictions, pattern_cts, variables=None):
        """ Makes the Balsubrani-Freund convex program like the new paper.
        """
        # set the comparison we want to use
        if self.use_equality_constraints:
            oper = operator.eq
        else:
            oper = operator.ge

        self._make_constraint_lhs_and_rhs(predictions, pattern_cts)

        if self.constraint == 'accuracy':
            c_rhs = self.cp_rhs_accs_
        elif self.constraint == 'class_accuracy':
            c_rhs = self.cp_rhs_joint_class_accs_
        else:
            c_rhs = self.cp_rhs_joint_conf_mat_

        # pad out the dimensions of c_rhs to deal with case where variables
        # is two dimensional
        c_rhs = np.expand_dims(c_rhs, axis=1)

        if variables is None:
            variables = cp.Variable((self.n_uniq_cols_ *\
                self.n_classes_, 1))

        constraints = []
        constraints.append(oper(self.c_ @ variables, c_rhs))

        # enforce that the predictions form a probability distribution
        constraints += [variables >= 0,
                cp.sum(variables, axis=0) == 1]

        # make the constraints that every model will have, i.e. enforcing
        # class frequencies and distribution of patterns.
        constraints += [cp.sum(cp.reshape(variables[:, i], \
                (self.n_uniq_cols_, self.n_classes_), order='C'), axis=1) ==\
                self.col_cts_ / self.n_samples_in_pred_x_
                for i in range(variables.shape[1])]
        constraints += [cp.sum(cp.reshape(variables[:, i], \
                (self.n_uniq_cols_, self.n_classes_), order='C'), axis=0) ==\
                self.class_freq_lb_
                for i in range(variables.shape[1])]

        return variables, constraints

    def predict(self, X, pattern_cts=None):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self)

        y = self.predict_proba(X, pattern_cts=pattern_cts)
        y = np.argmax(y, axis = 1)

        return self.le_.inverse_transform(y)

    def predict_proba(self, X, pattern_cts=None):
        """ Solves the convex program and returns the computed probabilities.

        """

        # Generate predictions for our new X if we have a known ensemble
        if self.unknown_ensemble:
            # make sure each estimator predicts on the same number of points
            if self.pred_type == 'determ':
                self.n_samples_in_pred_x_ = len(X[0])
                for i in range(1, self.n_estimators_):
                    if len(X[i]) != self.n_samples_in_pred_x_:
                        raise ValueError(f"Expected {self.n_samples_in_pred_x_}"
                                f" predictions, but estimator {i} only has"
                                f" {len(X[i])}.")
            elif self.pred_type == 'prob':
                self.n_samples_in_pred_x_ = X[0].shape[0]
                for i in range(1, self.n_estimators_):
                    if X[i].shape[0] != self.n_samples_in_pred_x_:
                        raise ValueError(f"Expected {self.n_samples_in_pred_x_}"
                                f"predictions, but estimator {i} only has"
                                f" {X[i].shape[0]}.")
            predictions = X

        else:
            self.n_samples_in_pred_x_ = X.shape[0]

            predictions = [
                    getattr(estimator, self.predict_method_)(X)
                    for estimator in self.estimators_
                    ]

        if self.prog_type == 'orig':
            pred_vars, constrs = self._make_orig_cvx_prog_constraints(predictions)
            self.cp_var_preds_ = pred_vars
            self.cp_pred_constraints_ = constrs
        else:
            pred_vars, constrs = self._make_joint_cvx_prog_constraints(predictions, pattern_cts)
            self.cp_var_joint_preds_ = pred_vars
            self.cp_pred_constraints_ = constrs

        if self.loss == '0-1':
            self.cp_objective_ = cp.Minimize(cp.sum(
                cp.norm(pred_vars, 'inf', axis=1)) / self.n_samples_in_pred_x_)
        elif self.loss == 'Xent':
            self.cp_objective_ = cp.Maximize(cp.sum(
                cp.entr(pred_vars)) / self.n_samples_in_pred_x_)

        problem = cp.Problem(self.cp_objective_, self.cp_pred_constraints_)

        # solve problem
        problem.solve(solver=self.solver, verbose=self.verbose)

        # retrieve and return solution
        probs = np.reshape(pred_vars.value, (self.n_uniq_cols_, self.n_classes_))

        # convert back from pattern to actual datapoints
        if self.prog_type == 'joint_compact':
            probs /= np.expand_dims(self.col_cts_, axis=1) / self.n_samples_in_pred_x_
            probs = probs[self.col_inv_, :]

        return probs

    def convert_gt_to_pattern(self, gt, joint_dist=True):
        """
        Encodes the ground truth in terms of patterns, rather than all
        datapoints.

        Conditional means the ground truth will be a conditional distribution
        of labels given the ensemble's prediction pattern.  Each row is
        associated with a pattern and sum to 1
        Joint will give the joint distribution of labels and patterns. The sum
        of all elemennts will sum to 1.
        """
        if not self._check_has_predicted_joint():
            raise AttributeError("You must predict with prog_type='joint_com\
                    act' before you can convert the ground truth.")

        if gt.ndim != 1:
            raise TypeError("Ground truth must be a 1D vector, but has"
                    f"{gt.ndim} dimensions and shape {gt.shape}.")
        gt = self.le_.transform(gt)
        gt_pattern = np.zeros((self.n_uniq_cols_, self.n_classes_))

        for i in range(self.n_uniq_cols_):
            curr_col = self.unique_cols_[:, i]
            offset = self.col_uniq_inds_[i]
            pred_subset = self.predictions_[:, offset:]
            inds = np.argwhere([np.array_equiv(pred_subset[:, j], curr_col)
                for j in range(pred_subset.shape[1])])
            for ind in inds:
                gt_pattern[i, gt[offset + ind]] += 1

        if not joint_dist:
            gt_pattern /= np.expand_dims(self.col_cts_, axis=1)
        else:
            gt_pattern /= np.sum(gt_pattern)

        return gt_pattern

    def get_confidence_intervals(self, joint_dist=True):
        """
        Gets confidence intervals for each pattern.  Does this by solving
        2* tau, where tau is the total number of patterns

        joint governs whether the confidence interval is for joint or
        conditional distribution
        """
        if not self._check_has_predicted_joint():
            raise AttributeError("You must predict with prog_type='joint_com\
                    act' before you can compute confidence intervals.")

        self.conf_int_ = np.zeros((self.n_uniq_cols_ * self.n_classes_, 2))

        #get constraints to form a program
        ci_vars, ci_constrs = self._make_joint_cvx_prog_constraints(self.predictions_)
        selection = cp.Parameter(self.n_uniq_cols_ * self.n_classes_)
        self.cp_var_ci_ = ci_vars
        self.cp_ci_constraints_ = ci_constrs

        lb_obj = cp.Minimize(selection.T @ ci_vars)
        ub_obj = cp.Maximize(selection.T @ ci_vars)

        prob_lb = cp.Problem(lb_obj, self.cp_ci_constraints_)
        prob_ub = cp.Problem(ub_obj, self.cp_ci_constraints_)

        def helper(prob, ind):
            basis_ind = np.zeros(selection.size)
            basis_ind[ind] = 1
            selection.value = basis_ind
            prob.solve(solver=self.solver, verbose=self.verbose, warm_start=True)

            return prob.value

        for i, prob in enumerate([prob_lb, prob_ub]):
            self.conf_int_[:, i] = Parallel(n_jobs=self.n_jobs)(
                    delayed(helper)(prob, ind)
                    for ind in range(self.n_uniq_cols_ * self.n_classes_))

        # do cleanup so we don't have negative lower bounds and upperbounds
        # that are greater than what's possible.
        pattern_dist = np.expand_dims(self.col_cts_ / np.sum(self.col_cts_), axis=1)
        #self.conf_int_[:, 0] = np.maximum(self.conf_int_[:, 0], 0)
        #self.conf_int_[:, 1] = np.minimum(self.conf_int_[:, 1].reshape(self.n_uniq_cols_, self.n_classes_), pattern_dist).reshape(self.n_uniq_cols_ * self.n_classes_)

        if not joint_dist:
            for i in range(2):
                self.conf_int_[:, i] = (self.conf_int_[:, i].reshape(self.n_uniq_cols_, self.n_classes_) / pattern_dist).reshape(self.n_uniq_cols_ * self.n_classes_)

        return self.conf_int_
