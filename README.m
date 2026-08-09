```python
import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import datetime
from openai import OpenAI


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Software TOPSIS - UTFPR",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# CSS SIMPLES
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3.5em;
    background-color: #003d7a;
    color: white;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0056a6;
    color: white;
}

.sidebar-card {
    background-color: gray;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #003d7a;
    margin-bottom: 20px;
}

.sidebar-card h4 {
    color: #003d7a;
    margin-top: 0;
}

.sidebar-card p {
    color: #222222;
    font-size: 14px;
}

/* Classe para forçar texto preto */
.custom-black {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNÇÃO PARA GERAR PDF
# ============================================================

def gerar_pdf(ranking, titulo, analise_ia=""):

    pdf = FPDF()
    pdf.add_page()

    # Cabeçalho
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        10,
        "Relatorio Detalhado TOPSIS",
        ln=True,
        align="C"
    )

    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        0,
        10,
        "UTFPR - Iniciacao Cientifica Ensino Medio",
        ln=True,
        align="C"
    )

    pdf.ln(5)

    # Informações
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        7,
        f"Analise: {titulo}",
        ln=True
    )

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        "Academico: Eduardo Fonseca Silveira | "
        f"Data: {datetime.datetime.now().strftime('%d/%m/%Y')}",
        ln=True
    )

    pdf.ln(5)

    # ========================================================
    # RANKING
    # ========================================================

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        10,
        "1. Ranking Final",
        ln=True
    )

    pdf.set_font("Helvetica", "", 9)

    pdf.cell(20, 8, "Rank", 1)
    pdf.cell(100, 8, "Alternativa", 1)
    pdf.cell(40, 8, "Score", 1)
    pdf.ln()

    for i, row in ranking.iterrows():

        pdf.cell(
            20,
            8,
            str(i + 1),
            1
        )

        pdf.cell(
            100,
            8,
            str(row["Alternativa"])[:45],
            1
        )

        pdf.cell(
            40,
            8,
            f"{row['Score TOPSIS']:.4f}",
            1
        )

        pdf.ln()

    # ========================================================
    # IA
    # ========================================================

    if analise_ia:

        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)

        pdf.cell(
            0,
            10,
            "2. Relatorio Interpretativo da IA",
            ln=True
        )

        pdf.set_font("Helvetica", "", 10)

        texto_limpo = (
            analise_ia
            .replace("**", "")
            .replace("#", "")
            .replace("•", "-")
            .encode("latin-1", "ignore")
            .decode("latin-1")
        )

        pdf.multi_cell(
            0,
            5,
            texto_limpo
        )

    # ========================================================
    # MEMÓRIA DE CÁLCULO
    # ========================================================

    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)

    pdf.cell(
        0,
        10,
        "3. Memoria de Calculo",
        ln=True
    )

    pdf.set_font("Helvetica", "", 9)

    pdf.multi_cell(
        0,
        5,
        "As matrizes de decisao normalizada, ponderada, "
        "solucoes ideais e distancias euclidianas foram "
        "calculadas conforme o metodo TOPSIS."
    )

    return bytes(pdf.output())


# ============================================================
# FUNÇÃO DA IA
# ============================================================

def gerar_analise_ia(
    api_key,
    ranking_df,
    pesos,
    criterios,
    tipos
):

    try:

        client = OpenAI(
            api_key=api_key
        )

        resumo = (
            f"Critérios avaliados: {', '.join(criterios)}\n"
        )

        resumo += (
            f"Pesos dos critérios: {pesos}\n"
        )

        resumo += (
            f"Tipos dos critérios: {tipos}\n\n"
        )

        resumo += (
            "Ranking final:\n"
        )

        resumo += ranking_df.to_string()

        prompt = f"""
 Você é um especialista em Análise de Decisão Multicritério (MCDA).

 Analise os resultados do método TOPSIS abaixo.

 {resumo}

 Produza um relatório interpretativo curto, direto e acadêmico.

 O relatório deve conter:

 1. Justificativa para a alternativa vencedora ter ficado em primeiro lugar,
 considerando os pesos dos critérios.

 2. Identificação do critério que mais contribuiu para o resultado.

 3. Uma conclusão ou recomendação prática.

 Escreva em português.

 Não seja excessivamente longo.

 Utilize Markdown simples.

 Não utilize emojis.
 """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente científico "
                        "especialista em tomada de decisão."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Erro ao conectar com a IA: {str(e)}"


# ============================================================
# ENGINE TOPSIS
# ============================================================

def executar_topsis(
    df,
    pesos,
    tipos,
    nomes_criterios
):

    matriz = (
        df
        .drop(columns=["Alternativa"])
        .values
        .astype(float)
    )

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    denominador = np.sqrt(
        (matriz ** 2).sum(axis=0)
    )

    denominador[
        denominador == 0
    ] = 1

    normalizada = (
        matriz / denominador
    )

    # --------------------------------------------------------
    # PONDERAÇÃO
    # --------------------------------------------------------

    ponderada = (
        normalizada *
        np.array(pesos)
    )

    # --------------------------------------------------------
    # SOLUÇÕES IDEAL E ANTI-IDEAL
    # --------------------------------------------------------

    ideal = []
    anti_ideal = []

    for i in range(len(tipos)):

        if tipos[i] == "lucro":

            ideal.append(
                np.max(
                    ponderada[:, i]
                )
            )

            anti_ideal.append(
                np.min(
                    ponderada[:, i]
                )
            )

        else:

            ideal.append(
                np.min(
                    ponderada[:, i]
                )
            )

            anti_ideal.append(
                np.max(
                    ponderada[:, i]
                )
            )

    ideal = np.array(ideal)
    anti_ideal = np.array(anti_ideal)

    # --------------------------------------------------------
    # DISTÂNCIAS
    # --------------------------------------------------------

    distancia_ideal = np.sqrt(
        ((ponderada - ideal) ** 2)
        .sum(axis=1)
    )

    distancia_anti = np.sqrt(
        ((ponderada - anti_ideal) ** 2)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # SCORE TOPSIS
    # --------------------------------------------------------

    scores = (
        distancia_anti /
        (
            distancia_ideal +
            distancia_anti +
            1e-12
        )
    )

    # --------------------------------------------------------
    # MATRIZES
    # --------------------------------------------------------

    matrizes = {

        "Normalizada":
            pd.DataFrame(
                normalizada,
                columns=nomes_criterios
            ),

        "Ponderada":
            pd.DataFrame(
                ponderada,
                columns=nomes_criterios
            ),

        "Ideais":
            pd.DataFrame(
                [ideal, anti_ideal],
                columns=nomes_criterios,
                index=[
                    "Ideal (+)",
                    "Anti-Ideal (-)"
                ]
            ),

        "Distâncias":
            pd.DataFrame({
                "S+ (Ideal)": distancia_ideal,
                "S- (Anti-Ideal)": distancia_anti
            })
    }

    return scores, matrizes


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("TOPSIS")

st.sidebar.markdown("---")

# Substituímos subheader por markdown com classe para permitir emoji e estilo preto
st.sidebar.markdown(
    '<p class="custom-black" style="font-weight:700;font-size:16px;margin:0;">🤖 Configuração da IA</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Texto separado do campo (agora usando a classe custom-black)
st.sidebar.markdown(
    """
    <p class="custom-black" style="font-weight:600;font-size:14px;margin-bottom:5px;">
        Insira sua OpenAI API Key
    </p>
    """,
    unsafe_allow_html=True
)

# Campo da API
api_key_input = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Cole sua chave aqui",
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Informações
st.sidebar.markdown(
    """
    <div class="sidebar-card">
", unsafe_allow_html=True)
