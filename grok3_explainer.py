# grok3_explainer.py

from typing import Dict


def generate_grok3_response(
    justificativa: str, porte: str, num_features: Dict[str, object], classe: str
) -> str:
    """
    Gera uma explicação textual simulada (estilo Grok-3) com base nos dados
    de entrada.

    Parâmetros:
        justificativa: texto fornecido pelo cliente.
        porte: porte da empresa (string).
        num_features: dicionário com chaves:
            - "anos_atividade" (int)
            - "rendimento_anual" (float)
            - "divida_total" (float)
        classe: string com a classe de risco prevista
            ("baixo", "médio" ou "alto").

    Retorna:
        Texto explicativo concatenando informações financeiras e textuais.
    """

    # Formatação de valores monetários para padrão brasileiro
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

    # Introdução comum
    texto = (
        f"A empresa foi classificada com risco **'{classe.upper()}'** com base em uma "
        f"análise integrada de dados financeiros, operacionais e "
        f"qualitativos. "
        f"A seguir, os principais "
        f"fatores considerados:\n\n"
        f"• **Porte da empresa**: sendo classificada como de porte *{porte}*, "
        f"o perfil de risco associado a esse segmento empresarial foi levado em conta.\n"
        f"• **Rendimento anual**: o valor de {rendimento_fmt} indica a capacidade de geração de "
        f"receita da empresa.\n"
        f"• **Tempo de atividade**: com {num_features['anos_atividade']} anos de operação, a "
        f"empresa demonstra um histórico de atuação relevante.\n"
        f"• **Dívida total**: o montante de {divida_fmt} fornece um indicativo do seu nível de "
        f"endividamento atual.\n"
        f'• **Justificativa textual**: a declaração "{justificativa.strip()}" foi considerada como '
        f"uma fonte complementar de evidências qualitativas.\n\n"
    )

    # Explicação final adaptativa por classe
    if classe.lower() == "baixo":
        texto += (
            "A classificação de **baixo risco** reflete uma combinação favorável de fatores, "
            "incluindo bom desempenho financeiro, estabilidade operacional e ausência de sinais de inadimplência. "
            "A justificativa apresentada também reforça a solidez e o comprometimento da empresa com suas obrigações financeiras."
        )
    elif classe.lower() == "médio":
        texto += (
            "A classificação de **risco médio** indica que, embora existam aspectos positivos como a experiência no mercado e capacidade de geração de receita, "
            "há também sinais de atenção, como nível de endividamento moderado ou justificativa com pontos de incerteza. "
            "A recomendação é manter o acompanhamento contínuo da empresa para identificar variações no perfil de risco."
        )
    elif classe.lower() == "alto":
        texto += (
            "A classificação de **alto risco** foi atribuída com base em indicadores que sugerem fragilidade financeira ou operacional. "
            "Fatores como alto nível de endividamento, baixo rendimento ou justificativa com informações limitadas ou preocupantes contribuíram significativamente para essa avaliação. "
            "Recomenda-se cautela em decisões de crédito relacionadas a essa empresa."
        )
    else:
        texto += (
            f"A classificação '{classe}' foi definida com base na análise dos dados disponíveis. "
            "Caso essa classe seja personalizada, recomenda-se complementar a explicação com informações adicionais."
        )

    return texto
