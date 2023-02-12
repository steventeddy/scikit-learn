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
import operator
import numpy as np
import cvxpy as cp

import warnings

from numpy.matlib import repmat
from scipy.sparse import coo_array
from statsmodels.stats.proportion import proportion_confint

from ._base import _BaseHeterogeneousEnsemble
from ._base import _fit_single_estimator
from ..base import ClassifierMixin
from ..base import clone

from ..metrics import accuracy_score, confusion_matrix

from ..preprocessing import LabelEncoder, OneHotEncoder
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
        'random_state': ['random_state'],
        'n_jobs': [None, Integral],
            }

    #pylint: disable=too-many-arguments
    def __init__(
        self,
        estimators,
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
        random_state=None,
        n_jobs=None,
        ):

        super().__init__(estimators=estimators)
        self.estimators=estimators
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
        check_X_y(X, y)
        self._validate_params()
        if self.pred_type == 'prob' and self.prog_type == 'joint_compact':
            raise ValueError(f"Program type {self.prog_type} must be used with\
                    deterministic predictions, i.e. pred_type='determ'")
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

        # Generate predictions from prefit models.
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
        if self.constraint == 'accuracy':
            num_samples = _num_samples(X)
        elif self.constraint == 'confusion_matrix':
            conf_mat_samples = repmat(self.class_freq_cts_,\
                        self.n_classes_, 1).T

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
                    [y.size] * self.n_classes_)
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

    def _make_orig_cvx_prog_constraints(self, predictions, n_samples):
        """ Makes the Balsubrani-Freund convex program a la the original paper.
        """
        # set the comparison we want to use
        if self.use_equality_constraints:
            oper = operator.eq
        else:
            oper = operator.gt

        self.cp_constraints_ = []

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
                cm_len = np.pow(self.n_classes_, 2)
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
            self.oh_le_.fit(np.arange(self.n_classes_))

        self.cp_var_preds_ = cp.Variable((n_samples, self.n_classes_))

        predictions = [
                self.oh_le_.transform(prediction.reshape(-1,1)).toarray()
                for prediction in predictions]

        for i in range(self.n_estimators_):
            if self.constraint == 'accuracy':
                self.cp_constraints_.append(oper(
                    cp.sum(
                        cp.multiply(predictions[i], self.cp_var_preds_)),
                    self.cp_rhs_accs_ * n_samples))

            elif self.constraint == 'class_accuracy':
                self.cp_constraints_.append(oper(
                    cp.sum(cp.multiply(predictions[i], self.cp_var_preds_)
                        , axis=0)
                    , self.cp_rhs_class_accuracies_[i]))
            else:
                lhs_ravel = np.ravel(self.cp_var_preds_.T\
                                                        @ predictions[i])
                rhs_ravel = np.ravel(self.cp_rhs_confusion_matrix_)
                self.cp_constraints_.append(oper(lhs_ravel, rhs_ravel))

        # make class frequency constraints
        self.cp_constraints_.append(oper(
            cp.sum(self.cp_var_preds_, axis=0),
            self.class_freq_lb_ * n_samples))

        # enforce each prediction being a probability distribution
        self.cp_constraints_.append(cp.sum(self.cp_var_preds_, axis=1)==1,
                self.cp_var_preds_ >= 0)

        return self

    def _make_joint_cvx_prog_constraints(self, predictions, n_samples):
        """ Makes the Balsubrani-Freund convex program like the new paper.
        """
        # set the comparison we want to use
        if self.use_equality_constraints:
            oper = operator.eq
        else:
            oper = operator.gt

        self.cp_constraints_ = []

        # transform the class accuracies/confusion matrix entries by
        # multiplying with class frequency lower bound.

        if self.constraint == 'class_accuracy':
            self.class_accuracy_joint_lb_ = [
                    np.multiply(class_accs, self.class_freq_lb_)
                    for class_accs in self.class_accuracy_lb_
                    ]
        elif self.constraint == 'confusion_matrix':
            class_freq_mat = repmat(self.class_freq_lb_, self.n_classes_, 1)
            self.confusion_matrix_joint_lb_ = [
                    np.multiply(conf_mat, class_freq_mat)
                    for conf_mat in self.confusion_matrix_lb_
                    ]

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
                    self.cp_rhs_joint_class_accs_[i * self.n_classes_ : (i + 1) * self.n_classes_].value = self.class_accuracy_joint_lb_[i])

            else:
                # number of entries in a confusion matrix
                cm_len = np.pow(self.n_classes_, 2)
                self.cp_rhs_joint_conf_mat_ = cp.Parameter(self.n_estimators_ *\
                        cm_len, nonneg=True)
                for i in range(self.n_estimators_):
                    self.cp_rhs_confusion_matrix_[i * cm_len : (i + 1) * cm_len] = np.ravel(self.confusion_matrix_joint_lb_[i])

        else:
            #set RHS to usual suspects
            self.cp_rhs_class_freq_ = self.class_freq_lb_

            if self.constraint == 'accuracy':
                self.cp_rhs_accs_ = self.accuracy_lb_
            elif self.constraint == 'class_accuracy':
                self.cp_rhs_joint_class_accs_ = self.class_accuracy_joint_lb_
            else:
                self.cp_rhs_joint_conf_mat_ = self.confusion_matrix_joint_lb_

        # convert predictions into 1 hot encoding and make the resulting
        # matrix sparse if deterministic predictions.
        #for each classifier, convert all their predictions to 1 hot
        # then append their predictions
        self.oh_le_ = OneHotEncoder(sparse_output=True)
        self.oh_le_.fit(np.arange(self.n_classes_))

        # compute the number of unique patterns
        predictions = np.array(predictions)
        unique_cols, col_inv, col_cts = np.unique(predictions, axis=1,\
                return_counts=True, return_inverse=True)
        # save the inverse so the prediction can be reconstructed
        self.joint_pred_col_inv = col_inv

        n_uniq_cols = unique_cols.shape[1]

        self.cp_var_joint_preds_ = cp.Variable((n_uniq_cols,\
                self.n_classes_))

        for i in range(self.n_estimators_):
            if self.constraint == 'accuracy':
                preds = unique_cols[i, :].reshape(-1, 1)
                preds_one_hot = self.oh_le_.transform(preds).to_array()
                preds_one_hot = np.reshape(preds_one_hot,\
                        (n_uniq_cols, self.n_classes_))
                self.cp_constraints_.append(oper(
                    cp.sum(cp.multiply(preds_one_hot, self.cp_var_joint_preds_)),
                    self.cp_rhs_accs_[i]))

            elif self.constraint == 'class_accuracy':
                # construct a sparse matrix for every classifier
                row = []
                col = []

                for j in range(n_uniq_cols):
                    row.append(unique_cols[i, j])
                    col.append(j * self.n_classes_ + unique_cols[i, j])

                row = np.array(row)
                col = np.array(col)
                data = [1] * len(row)
                C = coo_array((data, (row, col)),shape=(self.n_classes_, n_uniq_cols))
                self.cp_constraints_.append(oper(
                    C @ self.cp_var_joint_preds_.reshape(1, -1),
                    self.cp_rhs_joint_class_accs_[i]
                    ))

            else:
                # l ranges through the true classes
                for l in range(self.n_classes_):
                    # construct a sparse matrix for every classifier
                    row = []
                    col = []

                    for j in range(n_uniq_cols):
                        row.append(unique_cols[i, j])
                        col.append(j * self.n_classes_ + l)

                    row = np.array(row)
                    col = np.array(col)
                    data = [1] * len(row)
                    C = coo_array((data, (row, col)),shape=(self.n_classes_, n_uniq_cols))
                    self.cp_constraints_.append(oper(
                        C @ self.cp_var_joint_preds_.reshape(1, -1),
                        self.cp_rhs_joint_conf_mat_[i][l, :]
                        ))

        # make the constraints that every model will have, i.e. enforcing
        # class frequencies and distribution of patterns.
        self.cp_constraints_.append(
                cp.sum(self.cp_var_joint_preds_, axis=1) == col_cts/n_samples)
        self.cp_constraints_.append(oper(
                cp.sum(self.cp_var_joint_preds_, axis=0),
                self.class_freq_lb_))

        # enforce that the predictions form a probability distribution
        self.cp_constraints_.append(self.cp_var_joint_preds_ >= 0,
                cp.sum(self.cp_var_joint_preds_) == 1)

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
        check_is_fitted(self)

        y = self.predict_proba(X)
        y = np.argmax(y, axis = 0)
        return y

    def predict_proba(self, X):
        """ Solves the convex program and returns the computed probabilities.

        """
        n_samples = X.shape[0]

        # Generate predictions for our new X
        predictions = [
                getattr(estimator, self.predict_method_)(X)
                for estimator in self.estimators_
                ]
        if self.prog_type == 'orig':
            self._make_orig_cvx_prog_constraints(predictions, n_samples)
        else:
            self._make_joint_cvx_prog_constraints(predictions, n_samples)

        if self.loss == '0-1':
            self.cp_objective_ = cp.Minimize(cp.sum(cp.norm(self.cp_var_preds_, 'inf', axis=1)) / n_samples)
        elif self.loss == 'Xent':
            self.cp_objective_ = cp.Maximize(cp.sum(cp.entr(self.cp_var_preds_)) / n_samples)

        problem = cp.Problem(self.cp_objective_, self.cp_constraints_)

        # solve problem
        problem.solve(solver=self.solver, verbose=self.verbose)

        # retrieve and return solution
        probs = self.cp_var_preds_.value

        return probs
