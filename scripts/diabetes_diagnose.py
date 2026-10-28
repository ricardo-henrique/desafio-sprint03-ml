import sys

import joblib
import pandas as pd

# 1. Carregar os modelos e o scaler salvos anteriormente
try:
    model1 = joblib.load('model1_random_forest.pkl')
    model2 = joblib.load('model2_logistica.pkl')
    scaler = joblib.load('scaler_logistica.pkl')
    print("✅ Modelos e ferramentas carregados com sucesso!\n")
except FileNotFoundError:
    print("❌ Erro: Certifique-se de rodar o código de treino antes para gerar os arquivos .pkl.")
    sys.exit()

# 2. Criar os dados de um PACIENTE NOVO para teste
# NOTA: O modelo precisa exatamente das mesmas colunas (e na mesma ordem) usadas no treino.
# Para este exemplo, vamos criar um paciente fictício com exames alterados.

# Simulando a estrutura exata do seu dataset (exemplo simplificado com dados simulados)
# Substitua os valores abaixo pelos exames do paciente real que deseja testar:
dados_novo_paciente = {
    'age': [45.0],
    'bmi': [29.4],
    'fasting_glucose_mg_dl': [145.0],     # Glicose em jejum alta (comum em diabetes)
    'hba1c_percent': [7.2],               # Hemoglobina glicada alta (indicativo forte)
    'systolic_bp': [130.0],
    'diastolic_bp': [85.0],
    'total_cholesterol_mg_dl': [210.0],
    'family_history_diabetes': [1.0],     # Histórico familiar presente (1 para Sim, 0 para Não)
    'exercise_minutes_per_week': [45.0],
    'sleep_hours': [6.5]
}

# Transformando os dados em um DataFrame do Pandas
novo_paciente_df = pd.DataFrame(dados_novo_paciente)

# ----------------------------------------------------------------
# COMPATIBILIDADE DE COLUNAS: Garante que o formato bata com o treino
# ----------------------------------------------------------------
# Como o dataset original tem mais de 100 colunas, precisamos garantir que o DataFrame 
# do novo paciente tenha TODAS as colunas que o modelo espera (preenchendo com 0 as que faltarem)
colunas_treino = model1.feature_names_in_

for col in colunas_treino:
    if col not in novo_paciente_df.columns:
        novo_paciente_df[col] = 0.0

# Reordena as colunas para ficarem idênticas ao treino
novo_paciente_df = novo_paciente_df[colunas_treino]


# ================================================================
# EXECUTANDO OS DIAGNÓSTICOS
# ================================================================

# --- Diagnóstico do Model 1 (Random Forest) ---
# Random Forest aceita os dados diretos sem necessidade de escala
previsao_m1 = model1.predict(novo_paciente_df)[0]
# Também calcula a probabilidade (certeza) da escolha
probabilidade_m1 = model1.predict_proba(novo_paciente_df)[0]

# --- Diagnóstico do Model 2 (Regressão Logística) ---
# IMPORTANTE: A Regressão Logística EXIGE que passemos os dados pelo scaler primeiro
novo_paciente_scaled = scaler.transform(novo_paciente_df)
previsao_m2 = model2.predict(novo_paciente_scaled)[0]
probabilidade_m2 = model2.predict_proba(novo_paciente_scaled)[0]


# ================================================================
# EXIBIÇÃO DO PAINEL MÉDICO SIMULADO
# ================================================================
print("="*60)
print("             PAINEL DE DIAGNÓSTICO DA IA")
print("="*60)
print(f"Paciente avaliado: Idade {dados_novo_paciente['age'][0]} | Glicose: {dados_novo_paciente['fasting_glucose_mg_dl'][0]} mg/dL | HbA1c: {dados_novo_paciente['hba1c_percent'][0]}%")
print("-"*60)

# Resultado Model 1
print(f"🤖 [Model 1 - Random Forest] Diagnóstico: {previsao_m1}")
# Mostra o nível de certeza baseado na classe prevista
certeza_m1 = max(probabilidade_m1) * 100
print(f"   ↳ Nível de certeza do algoritmo: {certeza_m1:.2f}%")
print("-"*60)

# Resultado Model 2
print(f"📈 [Model 2 - Regressão Logística] Diagnóstico: {previsao_m2}")
certeza_m2 = max(probabilidade_m2) * 100
print(f"   ↳ Nível de certeza do algoritmo: {certeza_m2:.2f}%")
print("="*60)

if previsao_m1 == previsao_m2:
    print("\n🟢 Consenso: Ambos os modelos chegaram à mesma conclusão.")
else:
    print("\n🟡 Alerta: Os modelos divergiram no resultado. Analise as probabilidades!")
