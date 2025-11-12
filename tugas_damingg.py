import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("employee_data.csv")
X = df.drop(columns=["target"])   # ganti 'target' dengan nama kolom label kamu
y = df["target"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

with open("model_sklearn.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model sklearn berhasil disimpan!")
