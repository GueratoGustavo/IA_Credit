import os
import logging
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CreditPDFReport(FPDF):
    """
    Classe para geração de relatório em PDF com:
    - Classe prevista e probabilidades
    - Justificativa textual (LLM)
    - Gráfico SHAP explicativo
    """

    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Relatório de Predição de Risco - Empresa", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def add_prediction_page(
        self,
        idx: int,
        classe_predita: str,
        prob: np.ndarray,
        justificativa_text: str,
        shap_vals: np.ndarray,
        feature_names: List[str],
        output_dir: Optional[str] = None,
    ):
        """
        Adiciona uma página ao PDF com os dados da previsão de risco de crédito.

        Args:
            idx (int): Índice da amostra.
            classe_predita (str): Classe prevista (ex: "Alto Risco").
            prob (np.ndarray): Probabilidades de cada classe.
            justificativa_text (str): Texto gerado pela LLM explicando a decisão.
            shap_vals (np.ndarray): Valores SHAP da amostra.
            feature_names (List[str]): Nomes das features para o gráfico SHAP.
            output_dir (Optional[str]): Diretório para salvar a imagem temporária.
        """
        self.add_page()
        self.set_font("Arial", size=12)
        self.cell(
            0, 10, f"Empresa {idx + 1} - Classe Prevista: {classe_predita}", ln=True
        )
        self.cell(0, 10, f"Probabilidades: {np.round(prob, 3).tolist()}", ln=True)
        self.cell(0, 10, "Justificativa do Crédito (LLM):", ln=True)
        self.multi_cell(0, 10, justificativa_text)

        # Gerar e inserir o gráfico SHAP
        img_path = self._generate_shap_image(shap_vals, feature_names, idx, output_dir)
        if img_path:
            self.image(img_path, x=10, w=180)
            self._remove_temp_image(img_path)

    def _generate_shap_image(
        self,
        shap_vals: np.ndarray,
        feature_names: List[str],
        idx: int,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Gera e salva o gráfico SHAP como imagem temporária.

        Returns:
            Caminho da imagem gerada ou None em caso de erro.
        """
        try:
            output_dir = output_dir or os.getcwd()
            img_name = f"shap_plot_{idx}.png"
            img_path = os.path.join(output_dir, img_name)

            shap_values = shap.Explanation(
                values=shap_vals,
                base_values=np.zeros(len(shap_vals)),
                data=np.zeros(len(shap_vals)),
                feature_names=feature_names,
            )

            fig, ax = plt.subplots(figsize=(6, 3))
            shap.plots.bar(shap_values, show=False, max_display=10)
            plt.tight_layout()
            fig.savefig(img_path)
            plt.close(fig)
            return img_path

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico SHAP para idx {idx}: {e}")
            return None

    def _remove_temp_image(self, img_path: str):
        """Remove imagem temporária do SHAP plot."""
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            logger.warning(f"Erro ao remover imagem temporária: {e}")
