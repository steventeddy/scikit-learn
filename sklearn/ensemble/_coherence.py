"""Coherence Classification

This module contains a coherence classification method.

The module contains:

    - The `BalsubramaniFreundClassifier` implements coherence classification
    with an arbitrary ensemble of classifiers.
"""

# Author: Steven L. An <sla001@eng.ucsd.edu>
#
# License: BSD 3-Clause

from numbers import Integral, Real
import numpy as np
import cvxpy as cp

import warnings

from numpy.matlib import repmat
from statsmodels.stats.proportion import proportion_confint

from ._base import _BaseHeterogeneousEnsemble
from ._base import _fit_single_estimator
from ..base import ClassifierMixin
from ..base import clone

from ..metrics import accuracy_score, confusion_matrix

from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils import check_random_state, _safe_indexing
from ..utils.extmath import softmax
from ..utils.extmath import stable_cumsum
from ..metrics import accuracy_score, r2_score
from ..utils.parallel import delayed, Parallel
from ..utils.validation import check_is_fitted
from ..utils.validation import check_X_y
from ..utils.validation import has_fit_parameter
from ..utils.validation import _num_samples
from ..utils._param_validation import Interval, StrOptions
#from sklearn.utils.validation import check_array
#from sklearn.utils.multiclass import unique_labels

# pylint: disable=too-many-instance-attributes
class BalsubramaniFreundClassifer(_BaseHeterogeneousEnsemble, ClassifierMixin):
    """ The Balsubramani-Freund classifier.

    Parameters
    ----------
    estimators : list of (name, scikit-learn classifiers) pairs
        A list of classifiers that form the ensemble.
    loss : {'0-1', 'Xent'}, default='Xent'
        The loss function to be used with Balsubramani-Freund, 0-1 or cross
        entropy (aka maximum entropy problem).
    constraint : {'accuracy', 'class_accuracy', 'confusion_matrix'},\
                 default='accuracy'
        Determines whether the convex program constraints will be 'accuracy'
        (1 constraint per classifier), 'class_accuracy' (k constraints per
        classifier), or 'confusion_matrix' (k^2 constraints per classifier).
    prog_type : {'orig', 'joint_compact', "joint_full"},\
                 default='joint_compact'
        ``orig`` givesthe convex program presented by Balsubramani & Freund,
        which is good when you may increase classifiers but not datapoints,
        ``joint_compact`` gives the convex program where duplicate points,
        i.e. ones with the same ensemble predictions are removed, while
        ``joint_full`` constructs the convex program with all possible
        ensemble predictions, which can be expensive (exponential in number of
        estimators.)
    solve_dual : {True, False}, default=False
        Which type of convex program that will be solved, primal (False)
        program's solution are the predictions, dual (True) program's
        solutions are the weights.
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
    random_state : RandomState
        The current random number generator.
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
        'estimators': [list],
        'loss': [StrOptions({'0-1', 'Xent'})],
        'constraint': [StrOptions({'accuracy', 'class_accuracy', \
                'confusion_matrix'})],
        'prog_type': [StrOptions({'orig', 'joint_compact', 'joint_full'})],
        'solve_dual': ['boolean'],
        'prefit': ['boolean'],
        'lb_type': [StrOptions({'wilson', 'agresti_coull','beta',\
                'binom_est'}), None],
        'signif_lvl': [Interval(Real, 0, 1, closed='neither')],
        'pred_type': [StrOptions({'determ', 'prob'})],
        'use_parameters': ['boolean'],
        'solver': [StrOptions(set(cp.installed_solvers())), None],
        'verbose': ['boolean'],
        'random_state': ['random_state'],
        'n_jobs': [None, Integral],
            }

    #pylint: disable=too-many-arguments
    def __init__(
        self,
        estimators,
        loss='Xent',
        constraint='accuracy',
        prog_type='joint_compact',
        solve_dual=False,
        prefit=False,
        lb_type='wilson',
        signif_lvl=0.05,
        pred_type='determ',
        use_parameters=False,
        solver=None,
        verbose=False,
        random_state=None,
        n_jobs=None,
        ):

        super().__init__(estimators=estimators)
        self.estimators=estimators
        self.loss = loss
        self.constraint = constraint
        self.prog_type = prog_type
        self.solve_dual = solve_dual
        self.prefit = prefit
        self.lb_type = lb_type
        self.signif_lvl = signif_lvl
        self.pred_type = pred_type
        self.use_parameters = use_parameters
        self.solver = solver
        self.verbose = verbose
        self.random_state = random_state
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
                        f"Please change the estimator with {name} and"
                        f"index {i} or set pred_type to 'determ'."
                            )

        return names, all_estimators

    def _confint(self, count, nobs):
        return proportion_confint(count, nobs, alpha=self.signif_lvl,\
                method=self.lb_type)


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

        predict_method = 'predict' if self.pred_type=='determ'\
                else 'predict_proba'

        # Generate predictions from prefit models.
        predictions = [
                getattr(estimator, predict_method)(X)
                for estimator in all_estimators
                if estimator != 'drop'
                ]

        # calculate class frequencies
        self.num_classes_ = len(self.classes_)
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
        if self.constraint == 'accuracy':
            num_samples = _num_samples(X)
        elif self.constraint == 'confusion_matrix':
            conf_mat_samples = repmat(self.class_freq_cts_,\
                        self.num_classes_, 1).T

        # perform lower bounding if asked for
        if self.lb_type is None:
            self.class_freq_lb_ = self.class_freq_cts_ / y.size
            if self.constraint == 'accuracy':
                self.accuracy_lb_ = self.accuracy_counts_ / num_samples
            elif self.constraint == 'class_accuracy':
                self.class_accuracy_lb_ = [
                        class_acc_count / self.class_freq_cts_\
                        for class_acc_count in self.class_accuracy_counts]
            elif self.constraint == 'confusion_matrix':
                self.confusion_matrix_lb_ = [
                        conf_mat_count / conf_mat_samples\
                        for conf_mat_count in self.confusion_matrix_counts_]
        else:
            self.class_freq_lb_, _ = \
                    self._confint(self.class_freq_cts_, \
                    [y.size] * self.num_classes_)
            if self.constraint == 'accuracy':
                self.accuracy_lb_ = \
                        [self._confint(acc_count, num_samples)\
                        for acc_count in self.accuracy_counts_]
            elif self.constraint == 'class_accuracy':
                self.class_accuracy_lb_, _ = \
                        [self._confint(class_acc_count, self.class_freq_cts_)\
                        for class_acc_count in self.class_accuracy_counts]
            elif self.constraint == 'confusion_matrix':
                self.confusion_matrix_lb_, _ = \
                        [self._confint(conf_mat_count, conf_mat_samples)\
                        for conf_mat_count in self.confusion_matrix_counts_]

        return self

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
        check_is_fitted(self, ['X_', 'y_'])
