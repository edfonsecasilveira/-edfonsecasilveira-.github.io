import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Software TOPSIS - UTFPR",
    page_icon="⚖️",
    layout="wide"
)

# --- ESTILIZAÇÃO CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #003d7a;
        color: white;
        font-weight: bold;
    }
    .sidebar-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #003d7a;
        margin-bottom: 20px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-card h4 { color: #003d7a; margin-top: 0; }
    .sidebar-card p { color: #222222; font-size: 14px; margin-bottom: 5px; font-weight: 500; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO PARA GERAR PDF ---
def gerar_pdf(ranking, matrizes, titulo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, "Relatorio Detalhado TOPSIS", ln=True, align='C')
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(200, 10, "UTFPR - Iniciacao Cientifica Ensino Medio", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 7, f"Analise: {titulo}", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 7, f"Academico: Eduardo Fonseca Silveira | Data: {datetime.datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "1. Ranking Final", ln=True)
    pdf.set_font("Helvetica", '', 9)
    pdf.cell(20, 8, "Rank", 1); pdf.cell(100, 8, "Alternativa", 1); pdf.cell(40, 8, "Score", 1); pdf.ln()
    for i, row in ranking.iterrows():
        pdf.cell(20, 8, str(i), 1)
        pdf.cell(100, 8, str(row['Alternativa']), 1)
        pdf.cell(40, 8, f"{row['Score TOPSIS']:.4f}", 1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
st.sidebar.image("https://portal.utfpr.edu.br/icones/cabecalho/logo-utfpr/@@images/image.png", width=180)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <h4>📌 Iniciação Científica</h4>
        <p><b>Instituição:</b> UTFPR</p>
        <p><b>Acadêmico:</b><br>Eduardo Fonseca Silveira</p>
        <hr style="margin: 10px 0;">
        <p style="font-size: 13px;"><b>Professoras Orientadoras:</b><br>• Fernanda C. Zola<br>• Daiane M. G. Chiroli</p>
    </div>
    """, unsafe_allow_html=True)

# --- ENTRADA DE DADOS ---
st.title("⚖️ Sistema de Apoio à Decisão - TOPSIS")

with st.expander("1. Configurações Iniciais", expanded=True):
    titulo_projeto = st.text_input("Título da Análise", "Minha Decisão Multicritério")
    c1, c2 = st.columns(2)
    qtd_c = c1.number_input("Qtd de Critérios", 1, 15, 3)
    qtd_a = c2.number_input("Qtd de Alternativas", 2, 50, 3)

st.header("2. Critérios e Pesos")

# AVISO DE NORMALIZAÇÃO
st.warning("⚠️ **Nota sobre os Pesos:** Independentemente dos valores inseridos, o sistema irá normalizá-los para que a soma total seja igual a 1 (100%). Isso garante que a importância relativa entre os critérios seja mantida matematicamente.")

nomes_crit, pesos_brutos, tipos = [], [], []
cols = st.columns(qtd_c)
for i in range(qtd_c):
    with cols[i]:
        n = st.text_input(f"C{i+1}", f"Critério {i+1}", key=f"nc_{i}")
        p = st.number_input(f"Peso C{i+1}", 0.0, 100.0, 1.0, step=0.01, format="%.2f", key=f"
