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
# ESTILIZAÇÃO
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
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #003d7a;
    margin-bottom: 20px;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
}

.sidebar-card h4 {
    color: #003d7a !important;
    margin-top: 0;
}

.sidebar-card p,
.sidebar-card span,
.sidebar-card b {
    color: #222222 !important;
    font-size: 14px;
    margin-bottom: 5px;
    font-weight: 500;
}

[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNÇÃO PARA GERAR PDF
# ============================================================

def gerar_pdf(ranking, matrizes, titulo, analise_ia=""):

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

    # Informações do projeto
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
        f"Academico: Eduardo Fonseca Silveira | "
        f"Data: {datetime.datetime.now().strftime('%d/%m/%Y')}",
        ln=True
    )

    pdf.ln(5)

    # Ranking
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

    # Análise da IA
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

    # Matrizes
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
# FUNÇÃO DA INTELIGÊNCIA ARTIFICIAL
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

        resumo_problema = (
            f"Critérios avaliados: {', '.join(criterios)}\n"
        )

        resumo_problema += (
            f"Pesos dos critérios: {pesos}\n"
        )

        resumo_problema += (
            f"Tipos dos critérios: {tipos}\n\n"
        )

        resumo_problema += (
            "Ranking final:\n"
        )

        resumo_problema += ranking_df.to_string()

        prompt = f"""
Você é um especialista em Análise de Decisão Multicritério (MCDA).

Analise os resultados do método TOPSIS apresentados abaixo.

{resumo_problema}

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

        return (
            f"Erro ao conectar com a IA: {str(e)}"
        )


# ============================================================
# ENGINE TOPSIS
# ============================================================

def executar_topsis_completo(
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

    # Normalização
    denominador = np.sqrt(
        (matriz ** 2).sum(axis=0)
    )

    # Evita divisão por zero
    denominador[
        denominador == 0
    ] = 1

    normalizada = (
        matriz / denominador
    )

    # Aplicação dos pesos
    ponderada = (
        normalizada * np.array(pesos)
    )

    # Soluções ideais
    v_pos = []
    v_neg = []

    for i in range(len(tipos)):

        if tipos[i] == "lucro":

            v_pos.append(
                np.max(
                    ponderada[:, i]
                )
            )

            v_neg.append(
                np.min(
                    ponderada[:, i]
                )
            )

        else:

            v_pos.append(
                np.min(
                    ponderada[:, i]
                )
            )

            v_neg.append(
                np.max(
                    ponderada[:, i]
                )
            )

    v_pos = np.array(v_pos)
    v_neg = np.array(v_neg)

    # Distância até o ideal
    s_pos = np.sqrt(
        ((ponderada - v_pos) ** 2)
        .sum(axis=1)
    )

    # Distância até o anti-ideal
    s_neg = np.sqrt(
        ((ponderada - v_neg) ** 2)
        .sum(axis=1)
    )

    # Coeficiente de proximidade
    scores = (
        s_neg /
        (s_pos + s_neg + 1e-12)
    )

    # Matrizes intermediárias
    intermediarias = {

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
                [v_pos, v_neg],
                columns=nomes_criterios,
                index=[
                    "Ideal (+)",
                    "Anti-Ideal (-)"
                ]
            ),

        "Distâncias":
            pd.DataFrame({
                "S+ (Ideal)": s_pos,
                "S- (Anti-Ideal)": s_neg
            })
    }

    return scores, intermediarias


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚖️ TOPSIS")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Inteligência Artificial")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Insira sua chave da API da OpenAI."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="sidebar-card">

    <h4>📌 Iniciação Científica</h4>

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
    "⚖️ Sistema de Apoio à Decisão - TOPSIS"
)

st.write(
    "Ferramenta para análise de decisão multicritério "
    "utilizando o método TOPSIS."
)


# ============================================================
# CONFIGURAÇÕES INICIAIS
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
    "💡 Utilize números decimais nos pesos, "
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
# NORMALIZAÇÃO DOS PESOS
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


# Mostrar pesos normalizados
with st.expander("⚖️ Ver pesos normalizados"):

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
# VISUALIZAÇÃO DA MATRIZ
# ============================================================

with st.expander("📋 Visualizar matriz de decisão"):

    st.dataframe(
        df_base,
        use_container_width=True
    )


# ============================================================
# EXECUTAR ANÁLISE
# ============================================================

st.markdown("---")

if st.button(
    "📊 EXECUTAR ANÁLISE COMPLETA"
):

    # Verificar pesos
    if soma_pesos <= 0:

        st.error(
            "A soma dos pesos precisa ser maior que zero."
        )

        st.stop()

    # Verificar nomes
    if len(set(nomes_crit)) != len(nomes_crit):

        st.error(
            "Os nomes dos critérios precisam ser diferentes."
        )

        st.stop()

    # Executar TOPSIS
    scores, matrizes = executar_topsis_completo(
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

    ranking.index += 1

    # Salvar sessão
    st.session_state["resultado"] = ranking
    st.session_state["matrizes"] = matrizes

    # ========================================================
    # RESULTADO
    # ========================================================

    st.balloons()

    st.success(
        f"🏆 Melhor escolha: "
        f"**{ranking.iloc[0]['Alternativa']}**"
    )

    st.subheader(
        "🏁 Ranking Final"
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
            "🤖 IA analisando os resultados..."
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
            "🤖 Relatório Interpretativo da IA"
        )

        st.markdown(
            analise
        )

    else:

        st.session_state[
            "analise_ia"
        ] = ""

        st.info(
            "ℹ️ Insira sua OpenAI API Key na barra lateral "
            "para gerar o relatório interpretativo."
        )

    # ========================================================
    # MATRIZES INTERMEDIÁRIAS
    # ========================================================

    st.subheader(
        "📂 Memória de Cálculo"
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
        "📥 Exportar Resultados"
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
            "📥 Baixar Ranking em CSV",
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
            st.session_state["matrizes"],
            titulo_projeto,
            analise_ia=texto_ia
        )

        st.download_button(
            "📄 Baixar Relatório em PDF",
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
