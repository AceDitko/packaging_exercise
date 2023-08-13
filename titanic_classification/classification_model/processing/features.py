from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract first letter of variable

        def __init__(self, variables):
                
            if not isinstance(variables, list):
                    raise ValueError('variables should be a list')
                
            self.variables = variables

        def fit(self, X, y=None):
            # We need this step to fit the sklearn pipeline
            return self
        
        def transform(self, X):
              
            # So that we do not over-write the original dataframe

            X = X.copy()

            for feature in self.variables:
                X[feature] = x[feature].str[0]

            return X

        
