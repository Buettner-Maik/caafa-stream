from joblib import Parallel
#from multiprocessing.sharedctypes import Value

from sklearn.compose import ColumnTransformer
from sklearn.utils import _safe_indexing

from sklearn.utils.fixes import delayed



class ColumnTransformerNoCloning(ColumnTransformer):
    def __init__(
        self, 
        transformers, 
        *, 
        remainder="drop", 
        sparse_threshold=0.3, 
        n_jobs=None, 
        transformer_weights=None, 
        verbose=False, 
        verbose_feature_names_out=True
    ):
        super().__init__(
            transformers=transformers, 
            remainder=remainder, 
            sparse_threshold=sparse_threshold, 
            n_jobs=n_jobs, 
            transformer_weights=transformer_weights, 
            verbose=verbose, 
            verbose_feature_names_out=verbose_feature_names_out
        )

    def _fit_transform(self, X, y, func, fitted=False, column_as_strings=False):
        """
        Private function to fit and/or transform on demand.

        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.
        """
        transformers = list(
            self._iter(
                fitted=fitted, replace_strings=True, column_as_strings=column_as_strings
            )
        )
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(
                    transformer=trans,
                    X=_safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname="ColumnTransformer",
                    message=self._log_message(name, idx, len(transformers)),
                )
                for idx, (name, trans, column, weight) in enumerate(transformers, 1)
            )
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(super()._ERR_MSG_1DCOLUMN) from e
            else:
                raise
