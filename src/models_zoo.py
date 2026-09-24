#Libraries
from dataclasses import dataclass, field
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

#optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True  
except ImportError:
    HAS_CAT = False
try:
    import torch  # noqa: F401 check if torch is installed
    from torchmlp import TorchMLPClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from config import RANDOM_STATE

'''wrapper for models requiring integer targets (labels to integers)'''
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base):
        self.base = base #them unmodified estimator so sklearn clone() working
        
    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder().fit(y) #learns sorted class vocabulary
        self.base.fit(X, self.le_.transform(y), **fit_params)
        self.classes_ = self.le_.classes_ #class names sklearn API expected
        return self
    
    '''predicts-maps the intgers back to original labels'''
    def predict(self, X):
        return self.le_.inverse_transform(self.base.predict(X).astype(int))
    
    '''probabilities for each class'''
    def predict_proba(self, X):
        return self.base.predict_proba(X)

#models support class_weight="balanced" for imbalanced datasets
CLASS_WEIGHT_CAPABLE = {"LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees", "LinearSVM", "LightGBM", "TorchMLP"}


'''describes a model (name, factory, slow, family, params)'''   
@dataclass
class ModelSpec:
    name: str
    build: callable #lambda fresh instance on every call
    slow: bool = False  #whether the model is slow to train
    family: str = "classic"  #the family to which the model belongs
    params: dict = field(default_factory=dict)  #hyperparameters used for the model 

'''Available tuned models'''    
def model_zoo(n_classes: int):
    zoo= [ModelSpec("LogisticRegression", lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=RANDOM_STATE),family = "linear", params={"C": 1.0, "max_iter": 2000}),
        ModelSpec("GaussianNB", lambda: GaussianNB(), family = "bayes", params={}),
        ModelSpec("DecisionTree", lambda: DecisionTreeClassifier(max_depth=25, min_samples_leaf=3, random_state=RANDOM_STATE), family = "tree", params={"max_depth": 25, "min_samples_leaf": 3}),
        #ensemble models
        ModelSpec("RandomForest", lambda: RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), family = "ensemble", params={"n_estimators": 200, "min_samples_leaf": 2}),
        ModelSpec("ExtraTrees", lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), family = "ensemble", params={"n_estimators": 200, "min_samples_leaf": 2}),
        ModelSpec("HistGradientBoosting", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE), family = "boosting", params={"max_iter": 200, "learning_rate": 0.1}),
        ModelSpec("AdaBoost", lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=RANDOM_STATE), slow=True, family = "boosting", params={"n_estimators": 100, "base_max_depth": 3}),
        #slow models
        ModelSpec("KNN", lambda: KNeighborsClassifier(n_neighbors=7, n_jobs=-1), slow=True, family = "instance", params={"n_neighbors": 7}),
        ModelSpec("LinearSVM", lambda: LinearSVC(C=1.0, max_iter=20000, random_state=RANDOM_STATE), slow=True, family = "svm", params={"C":1.0,"max_iter": 20000}),
        #neural networks with early stopping and max_iter is the ceiling 
        ModelSpec("MLP", lambda: LabelEncodedClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, batch_size=512, max_iter=300, early_stopping=True, random_state=RANDOM_STATE)), slow=True, family = "neural", params={"hidden_layer_sizes": (128, 64), "max_iter": 300, "early_stopping": True}),
        ModelSpec("DeepMLP", lambda: LabelEncodedClassifier(MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-4, batch_size=512, max_iter=300, early_stopping=True, random_state=RANDOM_STATE)), slow=True, family = "neural", params={"hidden_layer_sizes": (256, 128, 64), "max_iter": 300, "early_stopping": True})]
    #optional models
    if HAS_XGB:
        zoo.append(ModelSpec("XGBoost", lambda: LabelEncodedClassifier(XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, subsample=0.9, colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss" if n_classes>2 else"logloss")), family = "boosting", params={"n_estimators": 300, "max_depth": 8, "learning_rate": 0.15}))
    if HAS_LGBM:
            zoo.append(ModelSpec("LightGBM", lambda: LGBMClassifier(n_estimators=300, num_leaves=64, learning_rate=0.1, reg_lambda=1.0, importance_type="gain", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1), family = "boosting", params={"n_estimators": 300, "num_leaves": 64, "learning_rate": 0.1, "reg_lambda": 1.0}))
    if HAS_CAT:
            zoo.append(ModelSpec("CatBoost", lambda: CatBoostClassifier(n_estimators=300, depth=8, learning_rate=0.15, verbose=0, random_state=RANDOM_STATE, allow_writing_files=False), family = "boosting", params={"iterations": 300, "depth": 8, "learning_rate": 0.15}))
    if HAS_TORCH:
            #the class default is bound at definition time
            zoo.append(ModelSpec("TorchMLP", lambda: TorchMLPClassifier(hidden=(256,128, 64), dropout=0.2, lr=1e-3, batch_size=1024, max_epochs=60, patience=6, random_state=RANDOM_STATE), slow=True, family = "neural", params={"hidden": (256, 128, 64), "dropout": 0.2, "lr": 1e-3, "max_epochs": 60, "early_stopping": True, "device": "cuda-if-available"}))
    return zoo


'''builds a model with class_weight="balanced" if supported by the model'''    
def build_with_class_weight(spec:ModelSpec):
    m=spec.build()
    #if wrapped, class_weight set on the inner model
    inner = m.base if isinstance(m, LabelEncodedClassifier) else m
    if spec.name in CLASS_WEIGHT_CAPABLE:
        try:
            inner.set_params(class_weight="balanced")
        except ValueError:
            pass
    return m 

            


