import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

class ShafiRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X, quantile=0.5):
    

        leaf_indices_test = self.rf.apply(X)
        
        leaf_indices_train = self.rf.apply(self.X_train)
        
        n_samples = X.shape[0]
        predictions = np.empty(n_samples)
       
        for i in range(n_samples):
            target_leaves = leaf_indices_test[i, :]
            
            matching_train_indices = np.where(leaf_indices_train == target_leaves)[0]
            
            y_values_in_leaves = self.y_train[matching_train_indices]
            
            predictions[i] = np.quantile(y_values_in_leaves, quantile)
            
        return predictions
    def save(self, filename="shafi_regressor.pkl"):
        """saving the modle in a .pkl file"""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model successfully saved to {filename}")

    @staticmethod
    def load(filename="quantile_rf_model.pkl"):
        """Loading saved pkl model"""
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model successfully loaded from {filename}")
        return model
