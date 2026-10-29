import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

file_name = 'data/diabetes_large_dataset.csv'
data = pd.read_csv(file_name)

column_target = 'diabetes_status'

data = data.dropna(subset=[column_target])

data[column_target] = data[column_target].astype(str)

remove_columns = [
    'patient_id', 'record_id', 'diabetes_type',
    'diabetes_medication', 'metformin_use', 'insulin_use', 
    'sulfonylurea_use', 'sglt2_inhibitor_use', 'glp1_medication_use', 
    'dpp4_inhibitor_use', 'previous_diabetes_diagnosis'
]

columns_to_remove = [col for col in remove_columns if col in data.columns]
data = data.drop(columns=columns_to_remove)

for col in data.columns:
    if col == column_target:
        continue

    numeric_column = pd.to_numeric(data[col], errors='coerce')

    if numeric_column.notna().sum() > (len(data)/2):
        data[col] = numeric_column
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].astype(str).fillna('Desconhecido')

X_raw = data.drop(column_target, axis=1)
y = data[column_target].copy()

# text_column = data.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
X_managed = pd.get_dummies(X_raw, drop_first=True)


X = X_managed.astype(float)

X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mantém a proporção das classes em cada fold da validação cruzada.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
}

# Compara os modelos nos dados de treino usando métricas ponderadas por classe.
models_for_cv = {
    'Random Forest (baseline)': RandomForestClassifier(random_state=42, n_estimators=100),
    'Regressão Logística': make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000),
    ),
}

print("\nVALIDAÇÃO CRUZADA STRATIFIED K-FOLD (5 folds)")
for model_name, estimator in models_for_cv.items():
    cv_results = cross_validate(
        estimator,
        X_training,
        y_training,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    print(f"\n{model_name}:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Espaço de busca dos hiperparâmetros testados para a Random Forest.
rf_param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_distributions,
    n_iter=10,
    scoring='f1_weighted',
    cv=cv,
    random_state=42,
    n_jobs=-1,
)

# Seleciona a combinação com melhor F1-score ponderado na validação cruzada.
print("\nAjustando hiperparâmetros da Random Forest...")
rf_search.fit(X_training, y_training)
model1 = rf_search.best_estimator_
print(f"Melhor F1-score ponderado na validação cruzada: {rf_search.best_score_:.4f}")
print(f"Melhores hiperparâmetros: {rf_search.best_params_}")

scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_test_scaled = scaler.transform(X_test)

print("Treinando o modelo 1 (Random Forest)...")
print("Usando o melhor estimador encontrado pela busca de hiperparâmetros.")

print("treinando o model 2 (Regressão Logística)...")
model2 = LogisticRegression(random_state=42, max_iter=1000)
model2.fit(X_training_scaled, y_training)

# previsões
pred_model1 = model1.predict(X_test)
pred_model2 = model2.predict(X_test_scaled)

# acurácia
acc_model1 = accuracy_score(y_test, pred_model1)
acc_model2 = accuracy_score(y_test, pred_model2)


print("\n" + "="*50)
print("             RESULTADO DA COMPARAÇÃO")
print("="*50)
print(f"Acurácia do Model 1 (Random Forest):      {acc_model1 * 100:.2f}%")
print(f"Acurácia do Model 2 (Regressão Logística): {acc_model2 * 100:.2f}%")
print("="*50 + "\n")

print("--- RELATÓRIO DETALHADO: MODEL 1 (Random Forest) ---")
print(classification_report(y_test, pred_model1))

print("\n--- RELATÓRIO DETALHADO: MODEL 2 (Regressão Logística) ---")
print(classification_report(y_test, pred_model2))

joblib.dump(model1, 'models/model1_random_forest.pkl')
joblib.dump(model2, 'models/model2_logistica.pkl')
joblib.dump(scaler, 'models/scaler_logistica.pkl')

print("Todos os modelos e o scaler foram salvos com sucesso!")