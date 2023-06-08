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
    prefit : {True, False}, default=False
        Whether or not the `estimators` provided have already been fitted.  If
        set to True, the estimators will not be refitted.
    bound_type : {'wilson', 'agresti_coull', 'beta', 'binom_test', None}\
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
        'prefit': ['boolean'],
        'bound_type': [StrOptions({'wilson', 'agresti_coull','beta',\
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
        prefit=False,
        bound_type='wilson',
        signif_lvl=0.05,
        pred_type='determ',
        use_parameters=False,
        solver=None,
        verbose=False,
        n_jobs=None,
        ):

        super().__init__(estimators=estimators)
        self.estimators = estimators
        self.unknown_ensemble = unknown_ensemble
        self.loss = loss
        self.constraint = constraint
        self.prefit = prefit
        self.bound_type = bound_type
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

    def _validate_params(self):
        super()._validate_params()

        if self.unknown_ensemble:
            warnings.warn(
                    "unknown_ensemble has been set to true.  Arguments for the"
                    "following parameters will be ignored: estimators, prefit,"
                    " bound_type, signif_level, n_job.  Make sure to use "
                    "fit_unknown_ensemble instead of fit."
                    )

        #change the normalization factor
        if not self.unknown_ensemble and self.estimators is None:
            raise ValueError("Must have non- None value for `estimators` "
                    "parameter when `unknown_ensemble`=False.")

        return self

    def _get_prediction_patterns(self, predictions, pattern_cts=None):
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
                method=self.bound_type)

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

        # perform bounding if asked for
        self.class_freq_est_ = self.class_freq_cts_ / y.size
        if self.constraint == 'accuracy':
            self.accuracy_est_ = self.accuracy_counts_ / num_samples
        elif self.constraint == 'class_accuracy':
            self.class_accuracy_est_ = [
                    class_acc_count / num_samples\
                    for class_acc_count in self.class_accuracy_counts]
        elif self.constraint == 'confusion_matrix':
            self.confusion_matrix_est_ = [
                    conf_mat_count / num_samples\
                    for conf_mat_count in self.confusion_matrix_counts_]

        if self.bound_type is not None:
            self.class_freq_lb_, self.class_freq_ub_ = \
                    self._confint(self.class_freq_cts_, \
                    [y.size] * self.n_classes_)
            if self.constraint == 'accuracy':
                self.accuracy_lb_, self.accuracy_ub_ = \
                    self._confint(self.accuracy_counts_, num_samples)
            elif self.constraint == 'class_accuracy':
                self.class_accuracy_lb_, self.class_accuracy_ub_ = \
                    self._confint(self.class_accuracy_counts, num_samples)
            elif self.constraint == 'confusion_matrix':
                self.confusion_matrix_lb_, self.confusion_matrix_ub_ = \
                    self._confint(self.confusion_matrix_counts_, num_samples)

        if self.constraint is ('class_accuracy' or 'confusion_matrix'):
            self.estimator_pred_class_freqs_ = [self.class_freq_lb_
                    for i in range(self.n_estimators_)]

        return self

    def fit_unknown_ensemble(
            self,
            estimator_params_lb,
            estimator_params_ub,
            class_freqs_lb,
            class_freqs_ub,
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
        if len(class_freqs_lb) != len(class_freqs_ub):
            raise ValueError('class_freqs_lb and class_freqs_ub have'
                    'different lengths.')
        if self.n_classes_ != len(class_freqs_lb):
            raise ValueError(f'{self.n_classes_} classes, yet there are '
                    f'{len(class_freqs_lb)} many class frequencies.')
        if len(estimator_params_lb) != len(estimator_params_ub):
            raise ValueError('estimator_params_lb and estimator_params_ub'
                    'have different lengths.')
        if self.n_estimators_ != len(estimator_params_lb):
            raise ValueError(f'{self.n_estimators_} estimators, yet there are '
                    f'{len(estimator_params_lb)} many sets of estimator '
                    'parameters.')

        # no need to check when using accuracies since the check is the same
        # as above
        if self.constraint == 'class_accuracy':
            for i in range(self.n_estimators_):
                if len(estimator_params_lb[i]) != self.n_classes_ or \
                        len(estimator_params_ub[i]) != self.n_classes_:
                    raise ValueError(f"Estimator {i} has "
                            f"{len(estimator_params_lb[i])} lower bounded"
                            " class accuracies and "
                            f"{len(estimator_params_ub[i])} upper bounded"
                            "class accuracies, "
                            f"rather than the expected {self.n_classes_}")
        elif self.constraint == 'confusion_matrix':
            for i in range(self.n_estimators_):
                if estimator_params_lb[i].shape !=\
                        (self.n_classes_, self.n_classes_) or \
                        estimator_params_ub[i].shape !=\
                        (self.n_classes_, self.n_classes_):
                    raise ValueError(f"Estimator {i} has confusion matrix lower"
                            f"bound of size size {estimator_params_lb[i].shape}"
                            " and confusion matrix upper bound of size "
                            f"{estimator_params_ub[i].shape} "
                            f"rather than the expected ({self.n_classes_}, "
                            f"{self.n_classes_}).")

        self.class_freq_lb_ = class_freqs_lb
        self.class_freq_ub_ = class_freqs_ub

        if self.constraint == 'accuracy':
            self.accuracy_lb_ = estimator_params_lb
            self.accuracy_ub_ = estimator_params_ub
        elif self.constraint == 'class_accuracy':
            self.class_accuracy_lb_ = estimator_params_lb
            self.class_accuracy_ub_ = estimator_params_ub
        elif self.constraint == 'confusion_matrix':
            self.confusion_matrix_lb_ = estimator_params_lb
            self.confusion_matrix_ub_ = estimator_params_ub

        self._validate_cvx_program_params()

        return self

    def _make_primal_cp_rhs(self):
        """
        Make the right hand side constraints
        """
        if self.use_parameters:
            self.cp_rhs_class_freq_lb_ = cp.Parameter(self.n_classes_, nonneg=True)
            self.cp_rhs_class_freq_ub_ = cp.Parameter(self.n_classes_, nonneg=True)
            self.cp_rhs_class_freq_lb_.value = self.class_freq_lb_
            self.cp_rhs_class_freq_ub_.value = self.class_freq_ub_

        if self.constraint == 'accuracy':
            if self.use_parameters:
                self.cp_rhs_classifier_lb_ = cp.Parameter(self.n_estimators_, nonneg=True)
                self.cp_rhs_classifier_ub_ = cp.Parameter(self.n_estimators_, nonneg=True)
                self.cp_rhs_classifier_lb_.value = self.accuracy_lb_
                self.cp_rhs_classifier_ub_.value = self.accuracy_ub_
            else:
                self.cp_rhs_classifier_lb_ = self.accuracy_lb_
                self.cp_rhs_classifier_ub_ = self.accuracy_ub_

        elif self.constraint == 'class_accuracy':
            if self.use_parameters:
                self.cp_rhs_classifier_lb_ = cp.Parameter(self.n_estimators_ * self.n_classes_, nonneg=True)
                self.cp_rhs_classifier_ub_ = cp.Parameter(self.n_estimators_ * self.n_classes_, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_classifier_lb_[i * self.n_classes_ : (i + 1) * self.n_classes_].value = self.class_accuracy_lb_[i]
                    self.cp_rhs_classifier_ub_[i * self.n_classes_ : (i + 1) * self.n_classes_].value = self.class_accuracy_ub_[i]
            else:
                self.cp_rhs_classifier_lb_ = np.zeros(self.n_estimators_ * self.n_classes_)
                self.cp_rhs_classifier_ub_ = np.zeros(self.n_estimators_ * self.n_classes_)
                for i in range(self.n_estimators_):
                    self.cp_rhs_classifier_lb_[i * self.n_classes_ : (i + 1) * self.n_classes_] = self.class_accuracy_lb_[i]
                    self.cp_rhs_classifier_ub_[i * self.n_classes_ : (i + 1) * self.n_classes_] = self.class_accuracy_ub_[i]
        else:
            # number of entries in a confusion matrix
            cm_len = np.power(self.n_classes_, 2)
            if self.use_parameters:
                self.cp_rhs_classifier_lb_ = cp.Parameter(self.n_estimators_ *\
                        cm_len, nonneg=True)
                self.cp_rhs_classifier_ub_ = cp.Parameter(self.n_estimators_ *\
                        cm_len, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_classifier_lb_[i * cm_len : (i + 1) * cm_len].value = np.ravel(self.confusion_matrix_lb_[i])
                    self.cp_rhs_classifier_ub_[i * cm_len : (i + 1) * cm_len].value = np.ravel(self.confusion_matrix_ub_[i])
            else:
                self.cp_rhs_classifier_lb_ = np.zeros(self.n_estimators_ *\
                        cm_len)
                self.cp_rhs_classifier_ub_ = np.zeros(self.n_estimators_ *\
                        cm_len)
                for i in range(self.n_estimators_):
                    self.cp_rhs_classifier_lb_[i * cm_len : (i + 1) * cm_len] = np.ravel(self.confusion_matrix_lb_[i])
                    self.cp_rhs_classifier_ub_[i * cm_len : (i + 1) * cm_len] = np.ravel(self.confusion_matrix_ub_[i])

    def _make_primal_cp_constraints(self, variables, predictions):
        """
        Make primal convex program constraints
        """

        self._make_primal_cp_rhs()

        constraints = []
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


        for i in range(self.n_estimators_):
            prediction_mass = np.sum(predictions[i])
            # compute the confusion matrix constraint LHS
            conf_mat_var_sums = variables.T @ predictions[i]

            if self.constraint == 'accuracy':
                acc_var_sums = cp.trace(conf_mat_var_sums)
                constraints.append(acc_var_sums >= \
                    self.cp_rhs_classifier_lb_[i] * prediction_mass)
                constraints.append(acc_var_sums <= \
                    self.cp_rhs_classifier_ub_[i] * prediction_mass)

            elif self.constraint == 'class_accuracy':
                class_acc_var_sums = cp.diag(conf_mat_var_sums)
                constraints.append(class_acc_var_sums >=\
                    self.cp_rhs_classifier_lb_[i] * prediction_mass)
                constraints.append(class_acc_var_sums <=\
                        self.cp_rhs_classifier_ub_[i] * prediction_mass)
            else:
                # Convert the confusion matrix into a matrix dealing with
                # joint probabilities, Pr(Y, h_i(X)) since when confusion
                # matrix entries are estimated via labeled data, we are
                # estimating Pr(Y | h_i(X)).  This is done by multiplying the
                # lower bounded class frequencies, one class per row.
                cm_len = np.power(self.n_classes_, 2)
                lhs_ravel = cp.reshape(conf_mat_var_sums, (cm_len, 1), order='C')
                rhs_lb_ravel = cp.reshape(self.cp_rhs_classifier_lb_[i] *\
                        prediction_mass, (cm_len, 1), order='C')
                rhs_ub_ravel = cp.reshape(self.cp_rhs_classifier_ub_[i] *\
                        prediction_mass, (cm_len, 1), order='C')
                constraints.append(lhs_ravel >= rhs_lb_ravel)
                constraints.append(lhs_ravel <= rhs_ub_ravel)

        # make class frequency constraints
        constraints.append(cp.sum(variables, axis=0) >= \
            self.class_freq_lb_ * self.n_samples_in_pred_x_)
        constraints.append(cp.sum(variables, axis=0) <= \
            self.class_freq_ub_ * self.n_samples_in_pred_x_)

        # enforce each prediction being a probability distribution
        constraints += [cp.sum(variables, axis=1) == 1,
                variables >= 0]

        return constraints

    def predict(self, X):
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

        y = self.predict_proba(X)
        y = np.argmax(y, axis = 1)

        return self.le_.inverse_transform(y)

    def predict_proba(self, X):
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

        self.predictions_ = predictions
        self._get_prediction_patterns(self.predictions_)

        pred_vars = cp.Variable((self.n_samples_in_pred_x_, self.n_classes_))
        constrs = self._make_primal_cp_constraints(pred_vars, predictions)
        self.cp_var_preds_ = pred_vars
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
        self.objective_value_ = problem.value

        # retrieve and return solution
        probs = pred_vars.value

        return probs

    def gt_pattern_average(self, gt, expand=True):
        """
        Averages the label distribution according to ensemble patterns
        """

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

        gt_pattern /= np.expand_dims(self.col_cts_, axis=1)

        if expand:
            return gt_pattern[self.col_inv_, :]
        else:
            return gt_pattern

    def get_confidence_intervals(self, expand=False):
        """
        Gets confidence intervals for each pattern.  Does this by solving
        2* tau, where tau is the total number of patterns

        joint governs whether the confidence interval is for joint or
        conditional distribution
        """

        if self.pred_type != 'determ':
            raise AttributeError('ensemble must predict deterministically to'
                    'compute confidence intervals.')

        self.conf_int_compact_ = np.zeros((self.n_uniq_cols_ * self.n_classes_, 2))

        #get constraints to form a program
        ci_vars = cp.Variable((self.n_samples_in_pred_x_, self.n_classes_))
        ci_constrs = self._make_primal_cp_constraints(ci_vars, self.predictions_)
        selection = cp.Parameter(self.n_samples_in_pred_x_* self.n_classes_)
        self.cp_var_ci_ = ci_vars
        self.cp_ci_constraints_ = ci_constrs
        lb_obj = cp.Minimize(selection.T @ cp.reshape(ci_vars, (self.n_samples_in_pred_x_ * self.n_classes_, 1), order='C'))
        ub_obj = cp.Maximize(selection.T @ cp.reshape(ci_vars, (self.n_samples_in_pred_x_ * self.n_classes_, 1), order='C'))

        prob_lb = cp.Problem(lb_obj, self.cp_ci_constraints_)
        prob_ub = cp.Problem(ub_obj, self.cp_ci_constraints_)

        def helper(prob, ind, offset):
            pt_inds = self.col_inv_[self.col_inv_ == ind]
            val = np.zeros(self.n_samples_in_pred_x_ * self.n_classes_)
            val[pt_inds * self.n_classes_ + offset] = 1
            selection.value = val
            prob.solve(solver=self.solver, verbose=self.verbose, warm_start=True)
            return prob.value / self.col_cts_[ind]

        for i, prob in enumerate([prob_lb, prob_ub]):
            for l in range(self.n_classes_):
                for ind in range(self.n_uniq_cols_):
                    self.conf_int_compact_[ind * self.n_classes_ + l, i]= helper(prob, ind, l)
                # class_inds = np.zeros(self.n_samples_in_pred_x_ * self.n_classes_)
                # class_inds[np.arange(self.n_samples_in_pred_x_) * self.n_classes_ + l] = 1
                # self.conf_int_compact[class_inds, i] = Parallel(n_jobs=self.n_jobs)(
                #         delayed(helper)(prob, ind, l)
                #         for ind in range(self.n_uniq_cols_))

        # do cleanup so we don't have negative lower bounds and upperbounds
        # that are greater than what's possible.
        self.conf_int_compact_[:, 0] = np.maximum(self.conf_int_compact_[:, 0], 0)
        self.conf_int_compact_[:, 1] = np.minimum(self.conf_int_compact_[:, 1], 1)

        if expand:
            self.conf_int_ = np.zeros((self.n_samples_in_pred_x_ * self.n_classes_, 2))
            expand_col_inv = np.zeros(self.n_samples_in_pred_x_ * self.n_classes_)
            tmp1 = (self.col_inv_ * self.n_classes_).astype(int)
            tmp2 = (self.col_inv_ * self.n_classes_ + 1).astype(int)
            expand_col_inv[0::2] = tmp1
            expand_col_inv[1::2] = tmp2
            expand_col_inv = expand_col_inv.astype(int)

            self.conf_int_[:, 0] = self.conf_int_compact_[:,0][expand_col_inv]
            self.conf_int_[:, 1] = self.conf_int_compact_[:,1][expand_col_inv]

            return self.conf_int_
        else:
            return self.conf_int_compact_
