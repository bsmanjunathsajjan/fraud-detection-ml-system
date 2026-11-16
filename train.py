"""Train a baseline fraud classifier on reproducible synthetic data."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.2, 1.0, n)
    hour = rng.integers(0, 24, n)
    distance = rng.exponential(15, n)
    failures = rng.poisson(0.25, n)
    risk = (amount > 500)*1.2 + (hour < 5)*0.8 + (distance > 80)*1.1 + (failures > 2)*1.4
    y = rng.binomial(1, 1/(1+np.exp(-(risk-2.4))))
    return np.column_stack([amount, hour, distance, failures]), y

def train():
    X, y = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42))
    model.fit(Xtr,ytr)
    pred = model.predict(Xte)
    print(classification_report(yte,pred,zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(yte,model.predict_proba(Xte)[:,1]),4))
    joblib.dump(model,"fraud_model.joblib")
if __name__ == "__main__": train()
