# Análise de Risco de Crédito com Deep Learning, BERT, SHAP e Relatórios PDF

Este projeto implementa uma solução completa e moderna para **classificação de risco de crédito de empresas**, combinando dados financeiros estruturados, variáveis categóricas e texto livre (justificativas) por meio de técnicas avançadas de **Deep Learning**, **processamento de linguagem natural (NLP)** com BERT, **explicabilidade de modelos (SHAP)** e geração automática de relatórios PDF detalhados.

---

## 📋 Sumário

- [Análise de Risco de Crédito com Deep Learning, BERT, SHAP e Relatórios PDF](#análise-de-risco-de-crédito-com-deep-learning-bert-shap-e-relatórios-pdf)
  - [📋 Sumário](#-sumário)
  - [Sobre o Projeto](#sobre-o-projeto)
  - [Tecnologias e Requisitos](#tecnologias-e-requisitos)
- [Análise de Risco de Crédito com Deep Learning, BERT, SHAP e Relatórios PDF](#análise-de-risco-de-crédito-com-deep-learning-bert-shap-e-relatórios-pdf-1)
  - [Modelo pré-treinado](#modelo-pré-treinado)
  - [Visão Geral do Pipeline](#visão-geral-do-pipeline)
  - [Principais Componentes do Código](#principais-componentes-do-código)
  - [Execução e Resultados](#execução-e-resultados)
  - [Explicabilidade e Relatórios](#explicabilidade-e-relatórios)
  - [Como Rodar](#como-rodar)

---

## Sobre o Projeto

Este projeto visa automatizar a análise de risco de crédito de empresas integrando diversas fontes de dados:

- **Dados Estruturados:** informações financeiras como anos de atividade, rendimento anual e dívida total.  
- **Variáveis Categóricas:** porte da empresa.  
- **Dados Textuais:** justificativas de crédito em linguagem natural.

A abordagem técnica une:

- **BERT em português (neuralmind)** para extrair embeddings ricos das justificativas textuais.  
- **Rede neural profunda** com entradas heterogêneas (dados numéricos, categóricos e texto) ajustada via **Keras Tuner (Hyperband)**.  
- **SHAP** para explicabilidade, detalhando a contribuição de cada variável para a predição.  
- **Geração automatizada de relatório em PDF** contendo resultados, gráficos SHAP e explicações interpretáveis simuladas por modelo LLM (mock Grok 3).

---

## Tecnologias e Requisitos

- Python 3.8+  
- Bibliotecas principais:

```bash
pip install pandas numpy tensorflow keras-tuner transformers shap matplotlib fpdf scikit-learn

# Análise de Risco de Crédito com Deep Learning, BERT, SHAP e Relatórios PDF

---

## Modelo pré-treinado

- **neuralmind/bert-base-portuguese-cased** (BERT em português)

---

## Visão Geral do Pipeline

1. **Carregamento e pré-processamento:**  
   leitura do CSV, tratamento de dados faltantes, normalização e codificação.

2. **Extração de embeddings:**  
   tokenização e vetorização do texto com BERT (vetor CLS).

3. **Divisão treino/teste:**  
   com estratificação para manter proporção das classes.

4. **Balanceamento de classes:**  
   cálculo de pesos para lidar com desequilíbrio.

5. **Ajuste do modelo:**  
   busca de hiperparâmetros com Keras Tuner (Hyperband).

6. **Treinamento:**  
   com callbacks para EarlyStopping.

7. **Avaliação:**  
   métricas no conjunto de teste.

8. **Explicabilidade:**  
   geração de valores SHAP para amostras selecionadas.

9. **Relatório:**  
   criação automática de PDFs com gráficos e explicações interpretativas.

---

## Principais Componentes do Código

| Componente             | Função                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------|
| `load_and_prepare_data`| Lê o arquivo CSV, preenche valores faltantes, normaliza dados numéricos e codifica categorias. |
| `get_bert_embeddings`  | Gera embeddings BERT para as justificativas, retornando vetores fixos para cada texto.|
| `build_model`          | Constrói a arquitetura da rede neural com múltiplas entradas (dados, categoria, texto).|
| `model_predict`        | Função wrapper para realizar predições e integrar com SHAP.                           |
| `PDF`                  | Classe responsável pela geração do relatório PDF contendo predições, gráficos SHAP e textos. |
| `generate_grok3_response` | Mock que simula explicações textuais inteligentes (substituível por APIs LLM reais).|

---

## Execução e Resultados

- O modelo treina combinando embeddings textuais, features numéricas e categóricas, otimizando a classificação em 3 classes de risco (baixo, médio, alto).  
- Utiliza pesos de classe para contornar desequilíbrio dos dados.  
- O desempenho é avaliado em métricas como acurácia, recall e matriz de confusão.  
- A explicabilidade via SHAP permite interpretar quais variáveis mais influenciam a decisão para cada empresa avaliada.

---

## Explicabilidade e Relatórios

O sistema gera para cada amostra um parecer interpretativo, integrando:

- Justificativa textual original.  
- Influência do porte da empresa.  
- Principais atributos que aumentam ou diminuem o risco segundo SHAP.

O relatório PDF consolidado contém:

- Predição da classe de risco e probabilidades associadas.  
- Texto da justificativa e explicação gerada.  
- Gráficos detalhados de valores SHAP por empresa.

Exemplo de explicação simulada:

> "A justificativa apresenta coerência com o histórico financeiro. O porte pequeno da empresa, aliado à dívida elevada, aumenta o risco de inadimplência conforme indicado pelos principais fatores explicativos."

---

## Como Rodar

1. Coloque o arquivo `empresas_credito_200k_justificativa.csv` no diretório raiz do projeto.  
2. Execute o script principal:

```bash
python seu_script.py
