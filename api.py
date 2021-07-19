"The NGBoost library API"
# pylint: disable=too-many-arguments
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state, check_X_y

from ngboost.distns import (
    Bernoulli,
    ClassificationDistn,
    LogNormal,
    Normal,
    RegressionDistn,
)
from ngboost.distns.utils import SurvivalDistnClass
from ngboost.helpers import Y_from_censored
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from ngboost.ngboost import NGBoost
from ngboost.scores import LogScore

import numpy as np

class NGBoostIncrement(NGBoost, BaseEstimator):
    """
    NGBoost iterative class:
        builds from NGBoost but can iterate parameters with a different learning rate and different data size
    """
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.001,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."

        if not hasattr(
            Dist, "scores"
        ):  # user is trying to use a dist that only has censored scores implemented
            Dist = Dist.uncensor(Score)

        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )


    def fit_iteration(
        self,
        NGBoost,
        X,
        Y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        """
        Fits an increment NGBoost model to the data

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            Y                     : DataFrame object or List or numpy array of outcomes (n)
                                    in numeric format. Should be floats for regression and
                                    integers from 0 to K-1 for K-class classification
            X_val                 : DataFrame object or List or
                                    numpy array of validation-set predictors in numeric format
            Y_val                 : DataFrame object or List or
                                    numpy array of validation-set outcomes in numeric format
            sample_weight         : how much to weigh each example in the training set.
                                    numpy array of size (n) (defaults to 1)
            val_sample_weight     : how much to weigh each example in the validation set.
                                    (defaults to 1)
            train_loss_monitor    : a custom score or set of scores to track on the training set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            val_loss_monitor      : a custom score or set of scores to track on the validation set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            early_stopping_rounds : the number of consecutive boosting iterations during which
                                    the loss has to increase before the algorithm stops early.
                                    

        Output:
            A fit NGBRegressor object
        """

        if Y is None:
            raise ValueError("y cannot be None")

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=self.multi_output)

        self.n_features = X.shape[1]

        loss_list = []
        
        #TODO PIERRE - DEFINE FIRST AS VALIDATION AND OTHER ONE AS TRAINING

        self.fit_init_params_to_marginal(Y) # initial parameters to marginal
        _,params = NGBoost.pred_dist(X) # initial parameters of first model before iterative boosting model
        
            
        if X_val is not None and Y_val is not None:
            X_val, Y_val = check_X_y(
                X_val, Y_val, y_numeric=True, multi_output=self.multi_output
            )
            _,val_params = NGBoost.pred_dist(X_val) # initial guess from validation 
            val_loss_list = []
            best_val_loss = np.inf

        if not train_loss_monitor:
            train_loss_monitor = lambda D, Y, W: D.total_score(  # NOQA
                Y, sample_weight=W
            )

        if not val_loss_monitor:
            val_loss_monitor = lambda D, Y: D.total_score(  # NOQA
                Y, sample_weight=val_sample_weight
            )  # NOQA

        for itr in range(self.n_estimators):
            _, col_idx, X_batch, Y_batch, weight_batch, P_batch = self.sample(
                X, Y, sample_weight, params
            )
            self.col_idxs.append(col_idx)

            D = self.Manifold(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch, weight_batch)]
            loss = loss_list[-1]
            grads = D.grad(Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, weight_batch)
            scale = self.line_search(proj_grad, P_batch, Y_batch, weight_batch)

            # pdb.set_trace()
            params -= (
                self.learning_rate
                * scale
                * np.array([m.predict(X[:, col_idx]) for m in self.base_models[-1]]).T
            )

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= (
                    self.learning_rate
                    * scale
                    * np.array(
                        [m.predict(X_val[:, col_idx]) for m in self.base_models[-1]]
                    ).T
                )
                val_loss = val_loss_monitor(self.Manifold(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if val_loss < best_val_loss:
                    best_val_loss, self.best_val_loss_itr = val_loss, itr
                if (
                    early_stopping_rounds is not None
                    and len(val_loss_list) > early_stopping_rounds
                    and best_val_loss
                    < np.min(np.array(val_loss_list[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                        print(
                            f"== Best iteration / VAL{self.best_val_loss_itr} (val_loss={best_val_loss:.4f})"
                        )
                    break

            if (
                self.verbose
                and int(self.verbose_eval) > 0
                and itr % int(self.verbose_eval) == 0
            ):
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                )

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result["train"] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result["val"] = {metric: val_loss_list}
        return self
    
    def pred_param(self, NGboostregressor, X, max_iter=None):
        m, n = X.shape
        
        #params = np.ones((m, self.Manifold.n_params)) * self.init_params # initial distribution homoskeodatic
        _,params = NGboostregressor.pred_dist(X) # inital parameters before boosting increment
        for i, (models, s, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs)
        ):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X[:, col_idx]) for model in models]).T
            params -= self.learning_rate * resids * s
        return params
    
    def pred_dist(self, NGboostregressor, X, max_iter=None):
        """
        Predict the conditional distribution of Y at the points X=x

        Parameters:
            X         : DataFrame object or List or
                        numpy array of predictors (n x p) in numeric format.
            max_iter  : get the prediction at the specified number of boosting iterations

        Output:
            A NGBoost distribution object
        """

        X = check_array(X)

        if (
            max_iter is not None
        ):  # get prediction at a particular iteration if asked for
            dist   = self.staged_pred_dist(NGboostregressor, X, max_iter=max_iter)[-1]
            params = []
        else:
            params = np.asarray(self.pred_param(NGboostregressor,X, max_iter))
            dist   = self.Dist(params.T)
        return dist,params
    
    def staged_pred_dist(self, NGboostregressor, X, max_iter=None):
        """
        Predict the conditional distribution of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of NGBoost distribution objects, one per boosting stage up to max_iter
        """
        predictions = []
        m, n = X.shape
        params = NGboostregressor.predict(X)
        for i, (models, s, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs)
        ):
            resids = np.array([model.predict(X[:, col_idx]) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(
                np.copy(params.T)
            )  # if the params aren't copied, param changes with stages carry over to dists
            predictions.append(dists)
            if max_iter and i == max_iter:
                break
        return predictions
    
    def predict(self, NGboostregressor, X, max_iter=None):
        """
        Point prediction of Y at the points X=x

        Parameters:
            X         : DataFrame object or List or numpy array of predictors (n x p)
                        in numeric Format
            max_iter  : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of Y
        """

        X = check_array(X)
        dist,_ = self.pred_dist(NGboostregressor, X, max_iter=max_iter)
        return dist.predict()
    

class NGBRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost regression models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBRegressor object that can be fit.
    """

    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."

        if not hasattr(
            Dist, "scores"
        ):  # user is trying to use a dist that only has censored scores implemented
            Dist = Dist.uncensor(Score)

        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )

    def __getstate__(self):
        state = super().__getstate__()
        # Remove the unpicklable entries.
        if self.Dist.__name__ == "DistWithUncensoredScore":
            state["Dist"] = self.Dist.__base__
            state["uncensor"] = True
        return state

    def __setstate__(self, state_dict):
        if "uncensor" in state_dict.keys():
            state_dict["Dist"] = state_dict["Dist"].uncensor(state_dict["Score"])
        super().__setstate__(state_dict)


class NGBClassifier(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost classification models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates classification.
    Use this class if you want to predict an outcome that could take a discrete number of
    (unordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Bernoulli
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBClassifier object that can be fit.
    """

    def __init__(
        self,
        Dist=Bernoulli,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, ClassificationDistn
        ), f"{Dist.__name__} is not useable for classification."
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )

    def predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of P(Y=k|X=x). Will have shape (n, K)
        """
        return self.pred_dist(X, max_iter=max_iter).class_probs()

    def staged_predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of of the estimates of P(Y=k|X=x) of shape (n, K),
            one per boosting stage up to max_iter
        """
        return [
            dist.class_probs() for dist in self.staged_pred_dist(X, max_iter=max_iter)
        ]


class NGBSurvival(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost survival models.

    NGBSurvival is a wrapper for the generic NGBoost class that facilitates survival analysis.
    Use this class if you want to predict an outcome that could take an infinite number of
    (ordered) values, but right-censoring is present in the observed data.

     Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. LogNormal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBSurvival object that can be fit.
    """

    def __init__(
        self,
        Dist=LogNormal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):

        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."
        if not hasattr(Dist, "censored_scores"):
            raise ValueError(
                f"The {Dist.__name__} distribution does not have any censored scores implemented."
            )

        SurvivalDistn = SurvivalDistnClass(Dist)

        # assert issubclass(Dist, RegressionDistn), f'{Dist.__name__} is not useable for survival.'
        super().__init__(
            SurvivalDistn,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # Both of the below contain SurvivalDistn
        del state["Manifold"]
        state["_basedist"] = state["Dist"]._basedist
        del state["Dist"]
        return state

    def __setstate__(self, state_dict):
        # Recreate the object which could not be pickled
        state_dict["Dist"] = SurvivalDistnClass(state_dict["_basedist"])
        del state_dict["_basedist"]
        state_dict["Manifold"] = manifold(state_dict["Score"], state_dict["Dist"])
        self.__dict__ = state_dict

    def fit(self, X, T, E, X_val=None, T_val=None, E_val=None, **kwargs):
        """Fits an NGBoost survival model to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            T                     : DataFrame object or List or
                                    numpy array of times to event or censoring (n) (floats).
            E                     : DataFrame object or List or
                                    numpy array of event indicators (n).
                                    E[i] = 1 <=> T[i] is the time of an event, else censoring time
            T_val                 : DataFrame object or List or
                                    validation-set times, in numeric format if any
            E_val                 : DataFrame object or List or
                                    validation-set event idicators, in numeric format if any
        """

        X = check_array(X)

        if X_val is not None:
            X_val = check_array(X_val)

        return super().fit(
            X,
            Y_from_censored(T, E),
            X_val=X_val,
            Y_val=Y_from_censored(T_val, E_val),
            **kwargs,
        )
