# Análise de Risco de Crédito com Deep Learning, BERT, SHAP e Relatórios PDF

Este projeto realiza a classificação de risco de crédito de empresas a partir de dados financeiros estruturados, categóricos e texto (justificativas). Utiliza embeddings BERT para texto, um modelo neural ajustado via Keras Tuner, explicabilidade com SHAP e geração automática de relatório em PDF com gráficos.

---

## Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Requisitos](#requisitos)
- [Estrutura do Código](#estrutura-do-código)
- [Funções Principais](#funções-principais)
- [Pipeline de Execução](#pipeline-de-execução)
- [Detalhes Técnicos](#detalhes-técnicos)
- [Geração de Explicações](#geração-de-explicações)
- [Geração do Relatório PDF](#geração-do-relatório-pdf)
- [Como Rodar](#como-rodar)
- [Considerações Finais](#considerações-finais)

---

## Descrição do Projeto

Este script implementa uma solução completa para análise preditiva de risco de crédito, combinando dados quantitativos e qualitativos. Utiliza:

- **Dados estruturados**: Anos de atividade, rendimento anual, dívida total
- **Variável categórica**: Porte da empresa
- **Texto livre**: Justificativa do crédito

O modelo combina embeddings do BERT em português (neuralmind), dados numéricos padronizados e embeddings categóricos para criar uma rede neural profunda ajustada automaticamente via Keras Tuner (Hyperband).

Explicabilidade é feita com a biblioteca SHAP e relatórios PDF são gerados automaticamente, incluindo gráficos e explicações textuais simuladas por um modelo LLM (mock Grok 3).

---

## Requisitos

- Python 3.8+
- Instalar dependências:

```bash
pip install pandas numpy tensorflow keras-tuner transformers shap matplotlib fpdf scikit-learn
