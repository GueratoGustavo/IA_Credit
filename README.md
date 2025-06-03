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
```

- Modelo pré-treinado BERT em português:
  - `neuralmind/bert-base-portuguese-cased`

---

## Estrutura do Código

- `load_and_prepare_data`: carrega e processa os dados.
- `get_bert_embeddings`: extrai embeddings do BERT.
- `build_model`: define o modelo com hiperparâmetros.
- `model_predict`: wrapper para SHAP.
- `PDF`: classe que gera relatórios em PDF.
- `generate_grok3_response`: mock de explicação LLM.
- `main`: pipeline completo de treino, predição e relatório.

---

## Funções Principais

### `load_and_prepare_data(csv_path)`

- Lê o CSV, preenche valores nulos.
- Normaliza os dados numéricos.
- Codifica `porte_empresa` e `risco_credito`.

---

### `get_bert_embeddings(texts, tokenizer, model, max_len=64, batch_size=32)`

- Tokeniza as justificativas e extrai o vetor CLS.
- Retorna matriz de embeddings do BERT.

---

### `build_model(hp, num_portes, num_features_len, text_embedding_dim)`

- Cria modelo com 3 entradas:
  - Porte (Embedding)
  - Numéricos (Dense)
  - Texto BERT (Input)
- Saída: 3 classes de risco com softmax.

---

### `model_predict(data)`

- Prepara dados para explicar via SHAP.
- Retorna `model.predict`.

---

### `PDF`

- Gera relatórios com título, gráficos SHAP, justificativas e explicações.
- Método `add_prediction` adiciona página por empresa.

---

### `generate_grok3_response(...)` e `gerar_explicacao_llm(...)`

- Simula uma explicação textual inteligente (mock de LLM).
- Pode ser trocado por API real no futuro.

---

## Pipeline de Execução

1. **Leitura e preprocessamento**
2. **Extração dos embeddings com BERT**
3. **Divisão treino/teste com stratify**
4. **Cálculo dos pesos de classe**
5. **Hiperparâmetros com Keras Tuner**
6. **Treinamento com EarlyStopping**
7. **Avaliação no conjunto de teste**
8. **Geração de SHAP para amostras**
9. **Criação do PDF com gráficos e textos**

---

## Detalhes Técnicos

- Vetor CLS do BERT representa justificativas.
- Camadas Dense com Dropout e BatchNorm.
- Pesos de classe balanceiam amostras desbalanceadas.
- Geração automática do gráfico SHAP por amostra.
- O mock de LLM pode ser substituído por Grok, GPT ou Claude via API.

---

## Geração de Explicações

Cada explicação textual simula um parecer técnico de crédito, combinando:

- Justificativa textual
- Porte da empresa
- Principais variáveis SHAP (positivas/negativas)

Exemplo:

> A justificativa apresentada demonstra coerência com o histórico financeiro. O porte pequeno, somado à dívida relativamente alta, eleva o risco, conforme indicado pelos principais atributos explicativos.

---

## Geração do Relatório PDF

O arquivo `relatorio_risco_empresas.pdf` inclui:

- Predição de risco (classe e probabilidades)
- Justificativa original
- Explicação textual gerada
- Gráfico SHAP para cada empresa

---

## Como Rodar

1. Adicione o CSV `empresas_credito_200k_justificativa.csv` ao diretório raiz
2. Execute:

```bash
python seu_script.py
```

3. O relatório `relatorio_risco_empresas.pdf` será gerado com as explicações.

---

## Considerações Finais

- Modular, fácil de expandir para mais dados ou variáveis
- Possível integração futura com:
  - LLMs reais via LangChain/OpenAI/xAI
  - Vetores históricos (FAISS/Chroma)
  - Painel visual com Streamlit ou Dash

---

## Código Fonte

O código completo está disponível no arquivo `seu_script.py`.

Se preferir, insira diretamente o conteúdo abaixo:

<pre><code>
# Cole aqui o código Python completo
</code></pre>

---
