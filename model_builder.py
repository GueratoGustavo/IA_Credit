from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Flatten,
    Concatenate,
    Dropout,
    BatchNormalization,
)


def build_model(
    hp, num_portes: int, num_features_len: int, text_embedding_dim: int
) -> Model:
    """
    Constrói o modelo Keras combinando:
        - Embedding do porte da empresa
        - Features numéricas (padronizadas)
        - Embedding textual de dimensão fixa
    Arquitetura definida via KerasTuner 
    (hiperparâmetros: número de layers, units, dropout, embedding_dim).
    """
    # Entradas     # Entradas\xa0
    entrada_porte = Input(shape=(1,), name="porte_input")
    entrada_num = Input(shape=(num_features_len,), name="num_input")
    entrada_text = Input(shape=(text_embedding_dim,), name="text_input")

    # Embedding para 'porte'
    emb_dim = hp.Int("embedding_dim", 4, 16, step=4)
    emb_porte = Embedding(
        input_dim=num_portes + 1, output_dim=emb_dim
    )(entrada_porte)
    porte_flat = Flatten()(emb_porte)

    # Concatenate: porte_flat + num_features + texto
    x = Concatenate()([porte_flat, entrada_num, entrada_text])

    # Camadas densas definidas pelo KerasTuner
    for i in range(hp.Int("n_layers", 1, 3)):
        units = hp.Int(f"units_{i}", 32, 256, step=32)
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float(f"dropout_{i}", 0.2, 0.6, step=0.1))(x)

    saida = Dense(3, activation="softmax")(x)
    model = Model(
        inputs=[entrada_porte, entrada_num, entrada_text],
        outputs=saida
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
