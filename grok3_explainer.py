from typing import Dict


def generate_grok3_response(
    justificativa: str, porte: str, num_features: Dict[str, object], classe: str
) -> str:
    """
    Gera uma explicação textual simulada (estilo Grok-3) com base nos dados de entrada.

    Parâmetros:
        justificativa: texto fornecido pelo cliente.
        porte: porte da empresa (string).
        num_features: dicionário com chaves:
            - "anos_atividade" (int)
            - "rendimento_anual" (float)
            - "divida_total" (float)
        classe: string com a classe de risco prevista.

    Retorna:
        Texto explicativo concatenando informações financeiras e textuais.
    """
    # Formata valores monetários para padrão brasileiro
    rendimento_fmt = (
        f"R$ {num_features['rendimento_anual']:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    divida_fmt = (
        f"R$ {num_features['divida_total']:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    texto = (
        f"A empresa foi classificada como '{classe}' com base nos seguintes fatores:\n"
        f"- Porte da empresa ({porte}): empresas de porte {porte} apresentam características que influenciam o risco de crédito.\n"
        f"- Rendimento anual ({rendimento_fmt}): esse valor sugere a capacidade financeira da empresa.\n"
        f"- Anos de atividade ({num_features['anos_atividade']} anos): indica o nível de experiência e estabilidade no mercado.\n"
        f"- Dívida total ({divida_fmt}): reflete o nível de endividamento da empresa.\n"
        f'- Justificativa fornecida: "{justificativa.strip()}" foi analisada, apontando fatores adicionais que corroboram a classificação.\n\n'
        f"Dessa forma, a classificação '{classe}' é justificada pela combinação de dados financeiros, operacionais e qualitativos."
    )
    return texto
