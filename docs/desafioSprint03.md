# Relatório — Desafio Sprint 03

## Introdução

Este relatório descreve o planejamento e o fluxo de trabalho de um projeto de aprendizado de máquina para classificação de dados relacionados a diabetes. O objetivo do protótipo é comparar modelos que estimam a classe indicada pela coluna `diabetes_status` a partir de atributos disponíveis no conjunto de dados.

O projeto foi desenvolvido em Python e executa localmente: prepara os dados, treina dois classificadores, apresenta métricas e salva os modelos para experimentação posterior. Os resultados são educacionais e não constituem diagnóstico, recomendação ou ferramenta clínica.

## Descrição do cenário escolhido

O cenário escolhido é a classificação de registros de pacientes em relação ao status de diabetes. O conjunto de dados utilizado é o `diabetes_large_dataset.csv`, obtido da fonte Kaggle indicada no README do projeto. A variável `diabetes_status` é o alvo; as demais colunas são tratadas como possíveis atributos preditores após a remoção de campos identificadores ou potencialmente associados a diagnóstico e tratamento prévios.

O problema é relevante para demonstrar um fluxo de classificação supervisionada e comparar abordagens com características diferentes. Porém, a qualidade do resultado depende da procedência, representatividade e qualidade dos dados. A presença de atributos clínicos ou sensíveis também exige cuidado com privacidade e controle de acesso. Antes de qualquer uso real, seria necessário validar a origem e a licença dos dados, avaliar vieses e realizar revisão por profissionais qualificados.

## Planejamento do ambiente de desenvolvimento

O desenvolvimento pode ser realizado em Windows, macOS ou Linux, utilizando Python e um ambiente virtual para isolar as dependências do projeto. O repositório já contém as dependências fixadas em `requirements.txt` e ignora ambientes virtuais, dados locais e modelos treinados por meio do `.gitignore`.

Fluxo sugerido para preparar e executar o projeto:

1. Clonar o repositório e abrir a pasta do projeto no VS Code.
2. Criar e ativar um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

   No PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   No Linux ou macOS:

   ```bash
   source .venv/bin/activate
   ```

3. Instalar as dependências listadas no arquivo:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Disponibilizar o CSV em `data/diabetes_large_dataset.csv`, respeitando as condições de acesso e uso da fonte.
5. Executar `python main.py` para treinar e avaliar os modelos. Os arquivos gerados são gravados em `models/`.
6. Usar os scripts da pasta `scripts/` para experimentar previsões e gráficos, observando que esses exemplos dependem dos modelos treinados e dos dados de entrada esperados.

O ambiente deve manter a mesma versão de Python e das dependências entre desenvolvimento, testes e execução. Como melhoria, recomenda-se documentar a versão de Python suportada e automatizar a configuração e os testes em integração contínua.

## Descrição das bibliotecas e fluxo de trabalho

As principais bibliotecas utilizadas são:

- **pandas**: leitura do CSV, seleção e transformação de colunas.
- **scikit-learn**: divisão entre treino e teste, padronização, treinamento e avaliação dos classificadores.
- **joblib**: serialização dos modelos e do scaler para uso posterior.
- **matplotlib** e **seaborn**: construção de gráficos de análise e interpretação.
- **numpy**: operações numéricas utilizadas nos gráficos.

O fluxo implementado em `main.py` começa pela leitura dos dados e remoção de linhas sem valor para o alvo. Em seguida, remove campos identificadores e colunas relacionadas a diagnóstico ou medicação prévia. As colunas com predominância de valores numéricos são convertidas para números e seus valores ausentes são preenchidos pela mediana; as demais são tratadas como categóricas, com valores ausentes representados como desconhecidos. As variáveis categóricas são convertidas em indicadores binários.

Os dados são divididos em conjuntos de treino e teste, com 20% reservados para teste e `random_state=42`. O projeto treina e compara:

- **Random Forest**, usando os atributos sem padronização;
- **Regressão Logística**, usando atributos padronizados com `StandardScaler`.

A comparação apresenta acurácia e relatório de classificação. Ao final, o Random Forest, a Regressão Logística e o scaler da regressão são salvos em `models/`. Os scripts gráficos exploram a importância das variáveis do Random Forest e os coeficientes da Regressão Logística.

Para uma avaliação mais confiável, etapas de pré-processamento que aprendem parâmetros dos dados — como imputação e definição das colunas categóricas — devem ser ajustadas apenas com o conjunto de treino, idealmente dentro de um `Pipeline` do scikit-learn. Também é recomendável avaliar métricas por classe, matriz de confusão, validação cruzada e calibração, além de registrar as versões do conjunto de dados e dos modelos. A acurácia isolada não é suficiente para concluir que um modelo é apropriado, especialmente em um cenário de saúde.

## Estratégia de controle de versão

O Git e o GitHub são utilizados para manter o histórico do código e facilitar a colaboração. Uma estratégia adequada para o projeto é:

- manter a branch principal estável e utilizar branches curtas para cada alteração, por exemplo, `feature/avaliacao-modelos` ou `docs/relatorio`;
- registrar mudanças em commits pequenos e descritivos, com mensagens no imperativo ou no padrão adotado pela equipe;
- abrir pull requests para revisão antes de integrar alterações à branch principal;
- executar verificações e testes antes do merge;
- criar tags para identificar versões estáveis e associá-las a uma versão do código, das dependências e do modelo.

O `.gitignore` exclui `data/` e `models/`, o que evita incluir no Git o CSV e os modelos binários gerados localmente. Essa política deve ser mantida para dados sensíveis ou volumosos. Para reproduzir experimentos, registre metadados (fonte e versão do conjunto de dados, parâmetros, métricas e hash dos artefatos) sem expor dados protegidos. Caso seja necessário versionar artefatos, prefira um armazenamento ou registro de modelos com controle de acesso e retenção apropriados, em vez de adicioná-los diretamente ao repositório.

## Proposta de infraestrutura em nuvem

Como o projeto atualmente é um protótipo executado localmente, a infraestrutura abaixo é uma proposta para uma etapa futura, não um serviço já implantado:

1. **Armazenamento de dados**: utilizar um bucket privado no Amazon S3 para dados autorizados e artefatos, com criptografia, políticas de acesso mínimo, versionamento e registros de auditoria. Dados pessoais identificáveis devem ser removidos ou protegidos antes do envio, conforme a legislação e as permissões aplicáveis.
2. **Treinamento**: executar o pipeline em um contêiner versionado, inicialmente em uma tarefa sob demanda no Amazon ECS com AWS Fargate, ou em uma instância apropriada caso os requisitos de processamento justifiquem. A imagem pode ser armazenada no Amazon ECR.
3. **Registro de modelos**: manter modelos aprovados, métricas, parâmetros e a versão do conjunto de dados em um registro controlado, como Amazon SageMaker Model Registry, permitindo rastreabilidade entre treinamento e implantação.
4. **Implantação futura**: somente após validação técnica, avaliação de risco e aprovação de uso, publicar uma API privada por meio de um serviço gerenciado, com autenticação, autorização, limites de requisições e validação de entrada. O modelo deve ser apresentado como suporte experimental, nunca como substituto de avaliação clínica.
5. **Operação e segurança**: centralizar logs e métricas no Amazon CloudWatch, criar alertas para falhas e acompanhar mudanças na distribuição dos dados e no desempenho. Usar IAM com privilégio mínimo, gestão de segredos, redes privadas quando aplicável, backups e política de retenção.
6. **Automação**: configurar uma pipeline de CI/CD para executar testes, verificar dependências e publicar imagens somente após revisão e aprovação. O treinamento e a implantação devem ser etapas separadas, com possibilidade de reverter para uma versão previamente validada.

Essa arquitetura mantém o protótipo simples e permite evoluir gradualmente, sem presumir que o modelo esteja pronto para uso clínico ou que exista uma implantação em nuvem no estado atual do projeto.

## Conclusão

O projeto apresenta um fluxo introdutório de classificação supervisionada aplicado a dados relacionados a diabetes, incluindo preparação dos dados, comparação entre Random Forest e Regressão Logística e salvamento dos artefatos. O uso de um ambiente virtual, dependências fixadas e controle de versão contribui para a organização e a reprodutibilidade do desenvolvimento.

Como próximos passos, recomenda-se consolidar o pré-processamento em pipelines ajustados somente com os dados de treino, ampliar a avaliação além da acurácia, acrescentar testes automatizados e registrar versões e métricas dos experimentos. Qualquer disponibilização em nuvem ou aplicação em saúde exige validação independente, proteção rigorosa dos dados e supervisão profissional.
