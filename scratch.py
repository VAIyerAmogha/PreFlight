import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TestTrans(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return np.array(["a"])
        
t = TestTrans()
t.set_output(transform="pandas")
print("Done")
