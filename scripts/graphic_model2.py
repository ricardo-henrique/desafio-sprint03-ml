import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Carregar o Modelo 2 (Regressão Logística) salvo anteriormente
try:
    model2 = joblib.load('model2_logistica.pkl')
    print("✅ Model 2 (Regressão Logística) carregado com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivo 'model2_logistica.pkl' não encontrado.")
    sys.exit()

# ================================================================
# MUDANÇA ESSENCIAL: Pegar os coeficientes em vez de feature_importances_
# ================================================================
# Extrai a força de peso de cada variável (usamos o valor absoluto np.abs para ver o impacto geral)
coeficientes = np.abs(model2.coef_[0])
nomes_colunas = model2.feature_names_in_

# 3. Criar e ordenar o DataFrame
df_importancia_logistica = pd.DataFrame({
    'Exame / Característica': nomes_colunas,
    'Peso do Coeficiente': coeficientes
})

# Ordenar do maior impacto para o menor
df_importancia_logistica = df_importancia_logistica.sort_values(by='Peso do Coeficiente', ascending=False)

# Selecionar as TOP 15 para o gráfico
df_top15_logistica = df_importancia_logistica.head(15)

# 4. Configurar e desenhar o gráfico
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

grafico = sns.barplot(
    x='Peso do Coeficiente', 
    y='Exame / Característica', 
    data=df_top15_logistica, 
    palette='magma',  # Usando uma cor diferente para diferenciar do modelo 1
    hue='Exame / Característica',
    legend=False
)

plt.title('Top 15 Características Mais Importantes para o Diagnóstico de Diabetes\n(Baseado na Regressão Logística)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Peso / Impacto Absoluto do Coeficiente', fontsize=12)
plt.ylabel('Exame / Indicador Clínico', fontsize=12)

plt.tight_layout()
print("\nGerando o gráfico da Regressão Logística...")
plt.show()

# Salvar o gráfico como imagem na mesma pasta
plt.savefig('data/img/importancia_features_diabetes2.png', dpi=300)