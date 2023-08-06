from abc import abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import clone as sk_clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numpy as np
import pandas as pd

class AbstractFeaturePairImputer(TransformerMixin, BaseEstimator):
    def __init__(self, pair_window_size=25,
                 importance='best', rmse_norm='maxmin', 
                 include_simple=True, manual_fit=False):
        """
        Base class for the creation of a Feature Pair Imputer
        :param pair_window_size: the size of the windows storing the feature pairs
        :param importance: the method for creating the importance weights
        :param rmse_norm: the method by which the error values are normalized
        :param include_simple: whether the mean and mode models are added to the feature pair models
        :param manual_fit: if true the windows of the feature pair imputer will not
                           automatically update its contents when fit is called
                           Used to omit the automatic fitting process of sklearn
                           and duplicating values in the windows when fit_transform is called
                           Call add_examples manually
        """
        self.pair_window_size = pair_window_size
        self.include_simple = include_simple
        self.manual_fit = manual_fit

        if importance != 'reciprocal' and importance != 'sum_difference' and importance != 'best':
            raise ValueError(f'Undefined importance method "{importance}"')
        self.importance = importance

        if rmse_norm != None and rmse_norm != 'maxmin' and rmse_norm != 'mean' and rmse_norm != 'stddev' and rmse_norm != 'iq':
            raise ValueError(f'Undefined RMSE normalization method "{rmse_norm}"')
        self.rmse_norm = rmse_norm

        self.num_features = -1
        self.num_feature_pairs = {}
        self.feature_pairs = {}

        self.initialized = False
        self.f_name_to_index = {}
        self.f_index_to_name = {}

        self.imputation_rmses = None # normalized RMSE
        self.importances = None # translated RMSE into basic weights

    def _initialize(self, columns):
        self.num_features = len(columns)
        self.f_name_to_index = {f:i for i, f in enumerate(columns)}
        self.f_index_to_name = {i:f for i, f in enumerate(columns)}
        self.imputation_rmses = np.zeros((self.num_features, self.num_features)) 

        self.initialized = True

    @abstractmethod
    def _update_rmse(self, indices):
        pass

    @abstractmethod
    def add_examples(self, X, indices=None):
        pass

    def _update_importances(self):
        if self.importance == 'reciprocal':
            nrmses = self.imputation_rmses + np.identity(self.num_features) * 0.001 # ensure checks for 0 do not trigger
            if (nrmses == 0).any():
                nrmses = (nrmses + 0.001) ** -1
            else:
                nrmses = nrmses ** -1
        elif self.importance == 'sum_difference':
            reg_sums = self.imputation_rmses.sum(axis=0)
            nrmses = (reg_sums - self.imputation_rmses) / reg_sums
        elif self.importance == 'best':
            nrmses = self.imputation_rmses.max() - self.imputation_rmses + 1
        else:
            raise ValueError(f'Undefined importance method "{self.importance}"')
        self.importances = nrmses

    def fit(self, X, y=None):
        if not self.initialized:
            self._initialize(X.columns)
            self.add_examples(X)
        elif not self.manual_fit:
            self.add_examples(X)
        return self

class PairRegImputer(AbstractFeaturePairImputer):
    def __init__(self, pair_window_size=25,
                 importance='best', rmse_norm='maxmin', 
                 include_simple=True, manual_fit=False):
        """
        A simple Regression imputer that keeps the last pair_window_size
        feature pair values to train a 1 to 1 Ordinary Least Squares regressor
        When imputing a weighted average of the regressors is chosen
        :param pair_window_size: how many value pairs are stored for each feature pair
        :param importance: the function used to convert the regression RMSEs into
                           importance weights
                           'best', 'reciprocal', 'sum_difference' are valid
        :param rmse_norm: the method by which the individual feature RMSE are
                          normalized
                          'maxmin', 'mean', 'stddev', 'iq', None are valid
        :param include_simple: if True adds the simple model that imputes values
                               with the mean of the feature from the last batch
        :param manual_fit: if True only adds feature pairs to storage when
                           add_examples is manually called or no data has been
                           stored yet
        """
        super().__init__(pair_window_size=pair_window_size,
                         importance=importance,
                         rmse_norm=rmse_norm,
                         include_simple=include_simple,
                         manual_fit=manual_fit)
        
        self.model = LinearRegression()
        self.slopes = None
        self.intercepts = None

        self.corr_calculated = False
        self.correlations = None

    def _initialize(self, columns):#num_features, feature_names):
        super()._initialize(columns)

        self.slopes = np.zeros((self.num_features, self.num_features))
        self.intercepts = np.zeros((self.num_features, self.num_features))

        for i in range(self.num_features):
            for k in range(i, self.num_features):
                self.num_feature_pairs[(i, k)] = 0
                self.feature_pairs[(i, k)] = np.zeros((self.pair_window_size, 2))

        self.correlations = np.identity(self.num_features)
        self.corr_calculated = False

    def get_correlation_matrix(self):
        if not self.corr_calculated:
            for i in range(self.num_features): 
                for k in range(i+1, self.num_features):
                    num_pairs = self.num_feature_pairs[(i, k)]
                    corr = np.corrcoef(self.feature_pairs[(i, k)][:num_pairs].T)[0, 1] if num_pairs != 0 else 0
                    self.correlations[i, k] = corr
                    self.correlations[k, i] = corr

            self.corr_calculated = True

        return self.correlations

    def _update_rmse(self, indices):
        """
        updates rmse of model in both directions
        """
        i, k = indices
        num_pairs = self.num_feature_pairs[indices]
        predicted_k = self.regress(indices, self.feature_pairs[indices][:num_pairs, 0])
        true_vals_k = self.feature_pairs[indices][:num_pairs, 1]
        
        self.imputation_rmses[indices] = np.sqrt(((predicted_k - true_vals_k) ** 2).mean())

        # normalize RMSE
        if self.rmse_norm == 'maxmin':
            maxmin = true_vals_k.max() - true_vals_k.min()
            if maxmin != 0: self.imputation_rmses[indices] /= maxmin
        elif self.rmse_norm == 'mean':
            avg = true_vals_k.mean()
            if avg != 0: self.imputation_rmses[indices] /= avg
        elif self.rmse_norm == 'stddev':
            stddev = np.sqrt(true_vals_k.var())
            if stddev != 0: self.imputation_rmses[indices] /= stddev
        elif self.rmse_norm == 'iq':
            iq = np.quantile(true_vals_k, 0.75) - np.quantile(true_vals_k, 0.25)
            if iq != 0: self.imputation_rmses[indices] /= iq
        elif self.rmse_norm is None:
            pass
        else:
            raise ValueError(f'Undefined RMSE normalization method "{self.rmse_norm}"')

        predicted_i = self.regress((k, i), self.feature_pairs[indices][:num_pairs, 1])
        true_vals_i = self.feature_pairs[indices][:num_pairs, 0]

        self.imputation_rmses[(k, i)] = np.sqrt(((predicted_i - true_vals_i) ** 2).mean())

        # normalize RMSE
        if self.rmse_norm == 'maxmin':
            maxmin = true_vals_i.max() - true_vals_i.min()
            if maxmin != 0: self.imputation_rmses[(k, i)] /= maxmin
        elif self.rmse_norm == 'mean':
            avg = true_vals_i.mean()
            if avg != 0: self.imputation_rmses[(k, i)] /= avg
        elif self.rmse_norm == 'stddev':
            stddev = np.sqrt(true_vals_i.var())
            if stddev != 0: self.imputation_rmses[(k, i)] /= stddev
        elif self.rmse_norm == 'iq':
            iq = np.quantile(true_vals_i, 0.75) - np.quantile(true_vals_i, 0.25)
            if iq != 0: self.imputation_rmses[(k, i)] /= iq
        elif self.rmse_norm is None:
            pass
        else:
            raise ValueError(f'Undefined RMSE normalization method "{self.rmse_norm}"')

    def add_examples(self, X, indices=None):
        if indices is not None and X.shape[1] == 2:
            examples = X.shape[0]
            
            if self.pair_window_size < examples:
                # only keep most recent if more examples than space in window
                X = X[-self.pair_window_size:]
                examples = self.pair_window_size
            
            num_pairs = self.num_feature_pairs[indices]
            pairs = self.feature_pairs[indices]

            if self.pair_window_size - num_pairs < examples:
                # sliding window, forget stuff when necessary
                num_kept_pairs = self.pair_window_size - examples
                pairs[:num_kept_pairs] = pairs[num_pairs-num_kept_pairs:num_pairs]
                pairs[num_kept_pairs:] = X
                self.num_feature_pairs[indices] = self.pair_window_size

            else:
                # if window has space just add it
                pairs[num_pairs:num_pairs+examples] = X
                self.num_feature_pairs[indices] += examples
            
            num_pairs = self.num_feature_pairs[indices]
            i, k = indices
            # update regressor coefficients
            if i == k:
                self.intercepts[indices] = np.mean(pairs[:num_pairs, 0])
            else:
                self.model.fit(pairs[:, :1], pairs[:, 1])
                self.slopes[indices] = self.model.coef_[0]
                self.intercepts[indices] = self.model.intercept_
                
                self.model.fit(pairs[:, 1:], pairs[:, 0])
                self.slopes[(k, i)] = self.model.coef_[0]
                self.intercepts[(k, i)] = self.model.intercept_

            self._update_rmse(indices)

        else:
            for i in range(self.num_features): 
                for k in range(i, self.num_features):
                    pairs = X.iloc[:, [i, k]].loc[(X.iloc[:, i].notna()) & (X.iloc[:, k].notna())]
                    self.add_examples(pairs, (i, k))
            self._update_importances()

        self.corr_calculated = False

    def _print_state(self):
        print(f"num_features\n{self.num_features}")
        print(f"pair_window_size\n{self.pair_window_size}")
        print(f"stored_pairs\n{self.num_feature_pairs}")
        print(f"feature_pairs\n{self.feature_pairs}")
        print(f"slopes\n{self.slopes}")
        print(f"intercepts\n{self.intercepts}")
        print(f"NRMSEs\n{self.imputation_rmses}")

    def regress(self, indices, X):
        return self.slopes[indices] * X + self.intercepts[indices]

    def transform(self, X):
        knowns = ~np.isnan(X) #xi f

        weights = np.einsum('ab,bc->abc', knowns, self.importances) #xi fin fout
        if self.include_simple:
            weights += np.identity(self.num_features) * self.importances
        if self.importance == 'best':
            weights *= weights == np.expand_dims(weights.max(axis=1), 1)
        weight_sums = np.sum(weights, axis=1) #xi fout
        regressed_values = np.nan_to_num(np.einsum('bc,ab->abc', self.slopes, X)) + self.intercepts # xi fin fout
        weighted_regressed_values = (regressed_values * weights).sum(axis=1) / weight_sums #xi fout

        if isinstance(X, pd.DataFrame):
            result = X.to_numpy()
        else:
            result = X.copy()
        np.putmask(result, np.isnan(result), weighted_regressed_values)
        indices = np.where(np.isnan(result))
        #result[indices] = np.take(self.intercepts[], indices[1])
        result[indices] = np.take(np.nanmean(result, axis=0), indices[1])

        self._ = weights
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return result

class NumToCatClassifier(TransformerMixin, BaseEstimator):
    def __init__(self):
        """
        Very basic one feature one label classifier
        Performs a mean distance (KNN) to label an observation
        """
        self.classes = None
        self.means = None
    
    def fit(self, X, y=None):
        self.classes = np.unique(y)
        self.means = np.zeros(self.classes.shape)
        for i, label in enumerate(self.classes):
            self.means[i] = X[y == label].mean()
    
    def predict(self, X):
        if hasattr(X, "to_numpy"):
            x = X.to_numpy()
        else:
            x = X.copy()
        closest = np.abs(np.tile(self.means, (X.shape[0], 1)) - x).argmin(axis=1)
        result = np.zeros(x.shape).astype('object')
        for val in range(self.classes.shape[0]):
            result[closest == val] = self.classes[val]
        return result.flatten()

class CatToNumClassifier(TransformerMixin, BaseEstimator):
    def __init__(self):
        """
        Very basic one feature one value regressor
        Performs a mean prediction based on the feature
        Features not yet seen during prediction are imputed by the overall mean
        """
        self.mean = 0
        self.means = {}
        
    def fit(self, X, y=None):
        self.mean = 0
        self.means = {}
        unique_values = np.unique(X)
        for val in unique_values:
            self.means[val] = y[X[:, 0] == val].mean()
            self.mean += self.means[val]
        self.mean /= unique_values.shape[0]

    def predict(self, X):
        if hasattr(X, "to_numpy"):
            result = X.to_numpy()
        else:
            result = X.copy()
        for val in np.unique(X):
            if val in self.means: 
                result[result == val] = self.means[val]
            else: 
                result[result == val] = self.mean
        return result.flatten()

class CatToCatClassifier(TransformerMixin, BaseEstimator):    
    def __init__(self):
        """
        Very basic one feature one label classifier
        Performs a mode prediction based on the feature
        Features not yet seen during prediction are given the majority class
        """
        self.mode = ''
        self.modes = {}
    
    def fit(self, X, y=None):
        self.mode = ''
        self.modes = {}
        unique_values = np.unique(X)
        for val in unique_values:
            self.modes[val] = stats.mode(y[X[:, 0] == val])[0][0]
        self.mode = stats.mode(y)[0][0]
    
    def predict(self, X):
        if hasattr(X, "to_numpy"):
            result = X.to_numpy()
        else:
            result = X.copy()
        for val in np.unique(X):
            if val in self.modes:
                result[result == val] = self.modes[val]
            else:
                result[result == val] = self.mode
        return result.flatten()

class OneValueClassifier(TransformerMixin, BaseEstimator):
    def __init__(self, value=None):
        """
        Basic classifier that returns value when asked to predict
        """
        self.value = None
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return np.repeat(self.value, X.shape[0])

class FeaturePairImputer(AbstractFeaturePairImputer):
    # ENSURE THAT THE ORDER OF FEATURES REMAINS THE SAME!
    def __init__(self, num_cols, cat_cols, pair_window_size=25, 
                 importance='best', rmse_norm='maxmin',
                 num_num_model=LinearRegression(),
                 num_cat_model=NumToCatClassifier(),#NumToCatClassifier(),#DecisionTreeClassifier(max_depth=3),
                 cat_num_model=CatToNumClassifier(),#CatToNumClassifier(),#SimpleImputer(strategy='mean'),
                 cat_cat_model=CatToCatClassifier(),#CatToCatClassifier(),#DecisionTreeClassifier(max_depth=3),
                 include_simple=True, manual_fit=False):
        """
        A simple imputer that keeps the last pair_window_size feature pair values
        to train imputation models or esimators to predict one feature using
        another
        When imputing a weighted average of the regressors is chosen
        :param num_cols: the numerical columns
        :param cat_cols: the categorical columns
        :param pair_window_size: how many value pairs are stored for each feature pair
        :param importance: the function used to convert the regression RMSEs into
                           importance weights
                           'reciprocal', 'best', 'sum_difference' are valid
        :param rmse_norm: the method by which the individual feature RMSE are
                          normalized
                          'maxmin', 'mean', 'stddev', 'iq', None are valid
                          normalization is required if the rmse are supposed
                          to be used as information for acquisition decisions
        :param include_simple: if True adds the simple model that imputes values
                               with the mean of the feature from the last batch
        :param manual_fit: if True only adds feature pairs to storage when
                           add_examples is manually called or no data has been
                           stored yet
        """
        super().__init__(pair_window_size=pair_window_size,
                         importance=importance,
                         rmse_norm=rmse_norm,
                         include_simple=include_simple,
                         manual_fit=manual_fit)
        
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        self.models = {}

        # Model Class Templates
        self.num_num_model = num_num_model
        self.num_cat_model = num_cat_model
        self.cat_num_model = cat_num_model
        self.cat_cat_model = cat_cat_model

    def _initialize(self, columns):
        super()._initialize(columns)

        for i, col1 in enumerate(columns):
            for k, col2 in enumerate(columns):
                if i == k: 
                    if (i, k) not in self.models: 
                        self.num_feature_pairs[(i, k)] = 0
                        self.feature_pairs[(i, k)] = np.zeros((self.pair_window_size, 2)).astype('object')
                        self.models[(i, k)] = OneValueClassifier()
                    continue

                self.num_feature_pairs[(i, k)] = 0
                self.feature_pairs[(i, k)] = np.zeros((self.pair_window_size, 2)).astype('object')
                if col1 in self.num_cols:
                    if col2 in self.num_cols:
                        self.models[(i, k)] = sk_clone(self.num_num_model)
                        self.models[(k, i)] = sk_clone(self.num_num_model)
                    else:
                        self.models[(i, k)] = sk_clone(self.num_cat_model)
                        self.models[(k, i)] = sk_clone(self.cat_num_model)
                else:
                    if col2 in self.num_cols:
                        self.models[(i, k)] = sk_clone(self.cat_num_model)
                        self.models[(k, i)] = sk_clone(self.num_cat_model)
                    else:
                        self.models[(i, k)] = sk_clone(self.cat_cat_model)
                        self.models[(k, i)] = sk_clone(self.cat_cat_model)

    def _update_rmse(self, indices):
        """
        updates rmse of model in both directions
        """
        i, k = indices
        num_pairs = self.num_feature_pairs[indices]
        predicted_k = self._transform(indices, self.feature_pairs[indices][:num_pairs, 0].reshape((-1, 1)))
        true_vals_k = self.feature_pairs[indices][:num_pairs, 1]
        
        if self.f_index_to_name[k] in self.num_cols:
            self.imputation_rmses[indices] = np.sqrt(((predicted_k - true_vals_k) ** 2).mean())

            # normalize RMSE
            if self.rmse_norm == 'maxmin':
                maxmin = true_vals_k.max() - true_vals_k.min()
                if maxmin != 0: self.imputation_rmses[indices] /= maxmin
            elif self.rmse_norm == 'mean':
                avg = true_vals_k.mean()
                if avg != 0: self.imputation_rmses[indices] /= avg
            elif self.rmse_norm == 'stddev':
                stddev = np.sqrt(true_vals_k.var())
                if stddev != 0: self.imputation_rmses[indices] /= stddev
            elif self.rmse_norm == 'iq':
                iq = np.quantile(true_vals_k, 0.75) - np.quantile(true_vals_k, 0.25)
                if iq != 0: self.imputation_rmses[indices] /= iq
            elif self.rmse_norm is None:
                pass
            else:
                raise ValueError(f'Undefined RMSE normalization method "{self.rmse_norm}"')
        else:
            # normalize to make comparable with RMSE?
            if True:
                self.imputation_rmses[indices] = 1 - (predicted_k == true_vals_k).sum() / self.num_feature_pairs[indices]
            #elif False:
            #    self.regression_rmses[indices] = 1 - max(0, self.regression_rmses[indices] / stats.mode(true_vals_k)[1][0])


        predicted_i = self._transform((k, i), self.feature_pairs[indices][:num_pairs, 1].reshape((-1, 1)))
        true_vals_i = self.feature_pairs[indices][:num_pairs, 0]

        if self.f_index_to_name[i] in self.num_cols:
            self.imputation_rmses[(k, i)] = np.sqrt(((predicted_i - true_vals_i) ** 2).mean())

            # normalize RMSE
            if self.rmse_norm == 'maxmin':
                maxmin = true_vals_i.max() - true_vals_i.min()
                if maxmin != 0: self.imputation_rmses[(k, i)] /= maxmin
            elif self.rmse_norm == 'mean':
                avg = true_vals_i.mean()
                if avg != 0: self.imputation_rmses[(k, i)] /= avg
            elif self.rmse_norm == 'stddev':
                stddev = np.sqrt(true_vals_i.var())
                if stddev != 0: self.imputation_rmses[(k, i)] /= stddev
            elif self.rmse_norm == 'iq':
                iq = np.quantile(true_vals_i, 0.75) - np.quantile(true_vals_i, 0.25)
                if iq != 0: self.imputation_rmses[(k, i)] /= iq
            elif self.rmse_norm is None:
                pass
            else:
                raise ValueError(f'Undefined RMSE normalization method "{self.rmse_norm}"')        
        else:
            # normalize to make comparable with RMSE?
            if True:
                self.imputation_rmses[(k, i)] = 1 - (predicted_k == true_vals_k).sum() / self.num_feature_pairs[indices]
            #elif False:
            #    self.regression_rmses[(k, i)] = 

    def add_examples(self, X, indices=None):
        if indices is not None and X.shape[1] == 2:
            examples = X.shape[0]
            
            if self.pair_window_size < examples:
                # only keep most recent if more examples than space in window
                X = X[-self.pair_window_size:]
                examples = self.pair_window_size
            
            num_pairs = self.num_feature_pairs[indices]
            pairs = self.feature_pairs[indices]

            if self.pair_window_size - num_pairs < examples:
                # sliding window, forget stuff when necessary
                num_kept_pairs = self.pair_window_size - examples
                pairs[:num_kept_pairs] = pairs[num_pairs-num_kept_pairs:num_pairs]
                pairs[num_kept_pairs:] = X
                self.num_feature_pairs[indices] = self.pair_window_size

            else:
                # if window has space just add it
                pairs[num_pairs:num_pairs+examples] = X
                self.num_feature_pairs[indices] += examples
            
            # update models
            num_pairs = self.num_feature_pairs[indices]
            i, k = indices

            if i == k:
                if self.f_index_to_name[i] in self.num_cols:
                    self.models[indices].value = np.mean(pairs[:num_pairs, 0])
                else:
                    self.models[indices].value = stats.mode(pairs[:num_pairs, 0])[0][0]
            else:
                self.models[indices].fit(pairs[:num_pairs, :1], pairs[:num_pairs, 1])
                
                self.models[(k, i)].fit(pairs[:num_pairs, 1:], pairs[:num_pairs, 0])

            self._update_rmse(indices)

        else:
            for i in range(self.num_features): 
                for k in range(i, self.num_features):
                    pairs = X.iloc[:, [i, k]].loc[(X.iloc[:, i].notna()) & (X.iloc[:, k].notna())]
                    self.add_examples(pairs, (i, k))
            self._update_importances()

    def _transform(self, indices, X):
        model = self.models[(indices)]
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            return model.transform(X)

    def _print_state(self):
        print(f"num_features\n{self.num_features}")
        print(f"pair_window_size\n{self.pair_window_size}")
        print(f"stored_pairs\n{self.num_feature_pairs}")
        #print(f"feature_pairs\n{self.feature_pairs}")
        print(f"NRMSEs\n{self.imputation_rmses}")

    def transform(self, X):
        # probably inefficient
        result = X.to_numpy()

        knowns = ~X.isna()

        weights = np.einsum('ab,bc->abc', knowns, self.importances) #xi fin fout
        if self.include_simple:
            weights += np.identity(self.num_features) * self.importances
        if self.importance == 'best':
            weights *= weights == np.expand_dims(weights.max(axis=1), 1)
        weight_sums = np.sum(weights, axis=1) #xi fout

        # flexibility of feature type requires individual decision making
        for i, col in enumerate(X):
            
            # separate between target feature being numbers or categories
            if col in self.num_cols: regressed = np.zeros(result.shape)
            else: regressed = np.zeros(result.shape).astype('object')

            for k, col2 in enumerate(X):
                # if i == k: regressed[:, k] = np.zeros(result.shape[0])
                # else: 

                # fill features to ensure no nan complaints from estimators
                # those data points get dropped eventually anyways
                if i < k: x = X.iloc[:, k:k+1].fillna(self.feature_pairs[(i, k)][0, 1])
                else: x = X.iloc[:, k:k+1].fillna(self.feature_pairs[(k, i)][0, 0])
                # if i == 1 and k == 0:
                #     print(X.iloc[:, :2])
                #     print(regressed[:, k])
                regressed[:, k] = self._transform((k, i), x)

            if col in self.num_cols:
                wei_regre = (regressed * weights[:, :, i]).sum(axis=1) / weight_sums[:, i]
                res = result[:, i].astype(float, casting='unsafe')
                np.putmask(res, np.isnan(res), wei_regre)

                # if complete nan instances occur do this failsafe
                indices = np.where(pd.isnull(res))
                res[indices] = np.nanmean(res, axis=0)
            else:
                best = np.take(regressed, weights[:, :, i].argmax(axis=1), axis=1)[:, 0] #xi
                res = result[:, i] # xi, 1 (col)
                np.putmask(res, pd.isnull(res), best)

            result[:, i] = res

        return pd.DataFrame(result, index=X.index, columns=X.columns)