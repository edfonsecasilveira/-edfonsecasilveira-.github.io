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

st.sidebar.subheader(
    "Configuração da IA"
)

# Texto separado do campo
st.sidebar.markdown(
    """
    <p style="
        color: #111111;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 5px;
    ">
        OpenAI API Key
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

    <h4>Iniciação Científica</h4>

    <p>
    <b>Instituição:</b> UTFPR
    </p>

    <p>
    <b>Acadêmico:</b><br>
    Eduardo Fonseca Silveira
    </p>

    <p>
    <b>Professoras Orientadoras:</b><br>
    Fernanda C. Zola<br>
    Daiane M. G. Chiroli
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "Sistema de Apoio à Decisão - TOPSIS"
)

st.write(
    "Ferramenta para análise de decisão multicritério "
    "utilizando o método TOPSIS."
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

with st.expander(
    "1. Configurações Iniciais",
    expanded=True
):

    titulo_projeto = st.text_input(
        "Título da Análise",
        "Minha Decisão Multicritério"
    )

    col1, col2 = st.columns(2)

    qtd_c = col1.number_input(
        "Quantidade de Critérios",
        min_value=1,
        max_value=15,
        value=3,
        step=1
    )

    qtd_a = col2.number_input(
        "Quantidade de Alternativas",
        min_value=2,
        max_value=50,
        value=3,
        step=1
    )


# ============================================================
# CRITÉRIOS
# ============================================================

st.header("2. Critérios e Pesos")

st.info(
    "Utilize números decimais nos pesos, "
    "como 1.50 ou 0.75."
)

nomes_crit = []
pesos_brutos = []
tipos = []

cols = st.columns(qtd_c)

for i in range(qtd_c):

    with cols[i]:

        nome = st.text_input(
            f"C{i + 1}",
            f"Critério {i + 1}",
            key=f"nome_criterio_{i}"
        )

        peso = st.number_input(
            f"Peso C{i + 1}",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            key=f"peso_{i}"
        )

        tipo = st.selectbox(
            f"Objetivo C{i + 1}",
            ["lucro", "custo"],
            key=f"tipo_{i}",
            help=(
                "Lucro: quanto maior, melhor. "
                "Custo: quanto menor, melhor."
            )
        )

        nomes_crit.append(nome)
        pesos_brutos.append(peso)
        tipos.append(tipo)


# ============================================================
# PESOS NORMALIZADOS
# ============================================================

soma_pesos = sum(
    pesos_brutos
)

if soma_pesos > 0:

    pesos_norm = [
        peso / soma_pesos
        for peso in pesos_brutos
    ]

else:

    pesos_norm = [
        1 / qtd_c
        for _ in range(qtd_c)
    ]


with st.expander(
    "Ver pesos normalizados"
):

    pesos_df = pd.DataFrame({
        "Critério": nomes_crit,
        "Peso Original": pesos_brutos,
        "Peso Normalizado": pesos_norm,
        "Tipo": tipos
    })

    st.dataframe(
        pesos_df,
        use_container_width=True
    )


# ============================================================
# MATRIZ DE DECISÃO
# ============================================================

st.header("3. Matriz de Decisão")

dados = []

for j in range(qtd_a):

    row = st.columns(
        [2] + [1] * qtd_c
    )

    nome_alternativa = row[0].text_input(
        f"Alternativa {j + 1}",
        f"Opção {j + 1}",
        key=f"alternativa_{j}"
    )

    valores = [
        nome_alternativa
    ]

    for i in range(qtd_c):

        valor = row[i + 1].number_input(
            nomes_crit[i],
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"valor_{j}_{i}"
        )

        valores.append(valor)

    dados.append(valores)


df_base = pd.DataFrame(
    dados,
    columns=["Alternativa"] + nomes_crit
)


# ============================================================
# VISUALIZAR MATRIZ
# ============================================================

with st.expander(
    "Visualizar matriz de decisão"
):

    st.dataframe(
        df_base,
        use_container_width=True
    )


# ============================================================
# EXECUTAR
# ============================================================

st.markdown("---")

if st.button(
    "EXECUTAR ANÁLISE COMPLETA"
):

    # Verificar pesos
    if soma_pesos <= 0:

        st.error(
            "A soma dos pesos precisa ser maior que zero."
        )

        st.stop()

    # Verificar critérios repetidos
    if len(set(nomes_crit)) != len(nomes_crit):

        st.error(
            "Os nomes dos critérios precisam ser diferentes."
        )

        st.stop()

    # TOPSIS
    scores, matrizes = executar_topsis(
        df_base,
        pesos_norm,
        tipos,
        nomes_crit
    )

    # Ranking
    resultado = df_base.copy()

    resultado["Score TOPSIS"] = scores

    ranking = (
        resultado
        .sort_values(
            by="Score TOPSIS",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Salvar na sessão
    st.session_state["resultado"] = ranking
    st.session_state["matrizes"] = matrizes

    # ========================================================
    # RESULTADO
    # ========================================================

    st.success(
        f"Melhor escolha: "
        f"**{ranking.iloc[0]['Alternativa']}**"
    )

    st.subheader(
        "Ranking Final"
    )

    st.dataframe(
        ranking.style.format({
            "Score TOPSIS": "{:.4f}"
        }),
        use_container_width=True
    )

    # ========================================================
    # IA
    # ========================================================

    if api_key_input:

        with st.spinner(
            "IA analisando os resultados..."
        ):

            analise = gerar_analise_ia(
                api_key_input,
                ranking,
                pesos_brutos,
                nomes_crit,
                tipos
            )

        st.session_state[
            "analise_ia"
        ] = analise

        st.subheader(
            "Relatório Interpretativo da IA"
        )

        st.markdown(
            analise
        )

    else:

        st.session_state[
            "analise_ia"
        ] = ""

        st.info(
            "Insira sua OpenAI API Key na barra lateral "
            "para gerar o relatório interpretativo."
        )

    # ========================================================
    # MEMÓRIA DE CÁLCULO
    # ========================================================

    st.subheader(
        "Memória de Cálculo"
    )

    with st.expander(
        "Matriz Normalizada"
    ):

        st.dataframe(
            matrizes["Normalizada"],
            use_container_width=True
        )

    with st.expander(
        "Matriz Ponderada"
    ):

        st.dataframe(
            matrizes["Ponderada"],
            use_container_width=True
        )

    with st.expander(
        "Soluções Ideais"
    ):

        st.dataframe(
            matrizes["Ideais"],
            use_container_width=True
        )

    with st.expander(
        "Distâncias Euclidianas"
    ):

        st.dataframe(
            matrizes["Distâncias"],
            use_container_width=True
        )


# ============================================================
# DOWNLOADS
# ============================================================

if "resultado" in st.session_state:

    st.markdown("---")

    st.subheader(
        "Exportar Resultados"
    )

    col1, col2 = st.columns(2)

    # CSV
    with col1:

        csv = (
            st.session_state["resultado"]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Baixar Ranking em CSV",
            data=csv,
            file_name="ranking_topsis.csv",
            mime="text/csv",
            use_container_width=True
        )

    # PDF
    with col2:

        texto_ia = st.session_state.get(
            "analise_ia",
            ""
        )

        pdf_out = gerar_pdf(
            st.session_state["resultado"],
            titulo_projeto,
            analise_ia=texto_ia
        )

        st.download_button(
            "Baixar Relatório em PDF",
            data=pdf_out,
            file_name="relatorio_topsis.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# RODAPÉ
# ============================================================

st.markdown("---")

st.caption(
    "Eduardo Fonseca Silveira | UTFPR | 2026"
)
```
