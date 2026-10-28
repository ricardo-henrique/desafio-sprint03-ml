import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carregar o Modelo 1 (Random Forest) salvo anteriormente
try:
    model1 = joblib.load('model1_random_forest.pkl')
    print("✅ Model 1 (Random Forest) carregado com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivo 'model1_random_forest.pkl' não encontrado. Execute o treino primeiro.")
    sys.exit()

# 2. Extrair a importância de cada variável
importancias = model1.feature_importances_
nomes_colunas = model1.feature_names_in_

# 3. Criar um DataFrame para organizar e ordenar os dados do maior para o menor
df_importancia = pd.DataFrame({
    'Exame / Característica': nomes_colunas,
    'Importância': importancias
})

# Ordenar pelas variáveis que mais pesaram na decisão da IA
df_importancia = df_importancia.sort_values(by='Importância', ascending=False)

# Como o seu dataset do Kaggle é gigante e tem mais de 100 colunas,
# vamos selecionar apenas as TOP 15 mais importantes para o gráfico ficar legível.
df_top15 = df_importancia.head(15)

# 4. Configurar e desenhar o gráfico
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Desenhar o gráfico de barras horizontais
grafico = sns.barplot(
    x='Importância', 
    y='Exame / Característica', 
    data=df_top15, 
    palette='viridis',
    hue='Exame / Característica',
    legend=False
)

# Adicionar títulos e rótulos ao gráfico
plt.title('Top 15 Características Mais Importantes para o Diagnóstico de Diabetes\n(Baseado no Modelo Random Forest)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Grau de Importância (Fator de Impacto)', fontsize=12)
plt.ylabel('Exame / Indicador Clínico', fontsize=12)

# Ajustar o layout para não cortar os nomes compridos das colunas
plt.tight_layout()

# 5. Exibir o gráfico na tela
print("\nGerando o gráfico... Uma janela deve se abrir na sua tela.")
plt.show()

# Salvar o gráfico como imagem na mesma pasta
plt.savefig('data/img/importancia_features_diabetes1.png', dpi=300)
