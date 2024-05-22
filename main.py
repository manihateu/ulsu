import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier

train_data = pd.read_parquet("train_data.pqt")
test_data = pd.read_parquet("test_data.pqt")

X = train_data.drop(columns=['end_cluster'])
y = train_data['end_cluster']

categorical_features = X.select_dtypes(include=['category', 'object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

y_one_hot = pd.get_dummies(y)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))])

X_train, X_valid, y_train, y_valid = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_valid)

test_probabilities = model.predict_proba(test_data)

submission_probabilities = pd.DataFrame({f'{i}': test_probabilities[i][:, 1] for i in range(len(test_probabilities))})

submission = pd.DataFrame(index=test_data.index)
for i, cluster in enumerate(["{other}", "{}", "{α, β}", "{α, γ}", "{α, δ}", "{α, ε, η}", "{α, ε, θ}", "{α, ε, ψ}", "{α, ε}", "{α, η}", "{α, θ}", "{α, λ}", "{α, μ}", "{α, π}", "{α, ψ}", "{α}", "{λ}"]):
    submission[cluster] = submission_probabilities[f'{i}']

submission.insert(0, 'id', test_data.index)

submission.to_csv('sample_submission.csv', index=False)
