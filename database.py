import pandas as pd

# Passo 1: Ler o CSV original
df = pd.read_csv("empresas_credito_200k_justificativa.csv")

# Passo 2: Embaralhar as linhas
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Passo 3: Salvar em um novo arquivo CSV
df_shuffled.to_csv("empresas_shuffled.csv", index=False)

print("Arquivo embaralhado salvo como 'empresas_shuffled.csv'.")
