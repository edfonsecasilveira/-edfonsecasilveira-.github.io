npm install firebase
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
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO PARA GERAR PDF ---
def gerar_pdf(df, titulo, autor):
    pdf = FPDF()
    pdf.add_page()
    
    # Cabeçalho Institucional
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Relatorio de Analise Multicriterio - TOPSIS", ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, "Iniciacao Cientifica Ensino Medio - UTFPR", ln=True, align='C')
    pdf.ln(10)
    
    # Informações do Projeto
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, f"Titulo: {titulo}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Academico: Eduardo Fonseca Silveira", ln=True)
    pdf.cell(0, 7, f"Professoras Responsaveis: Fernanda C. Zola e Daiane M. G. Chiroli", ln=True)
    pdf.cell(0, 7, f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(5)
    
    # Tabela de Resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Ranking de Alternativas:", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Cabeçalho da Tabela
    pdf.cell(20, 10, "Rank", 1)
    pdf.cell(100, 10, "Alternativa", 1)
    pdf.cell(40, 10, "Score", 1)
    pdf.ln()
    
    # Dados
    for i, row in df.iterrows():
        pdf.cell(20, 10, str(i), 1)
        pdf.cell(100, 10, str(row['Alternativa']), 1)
        pdf.cell(40, 10, f"{row['Score TOPSIS']:.4f}", 1)
        pdf.ln()
        
    return bytes(pdf.output())

# --- SIDEBAR ---
st.sidebar.image("https://portal.utfpr.edu.br/icones/cabecalho/logo-utfpr/@@images/image.png", width=180)
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Iniciação Científica")

# Cabeçalho Atualizado na Sidebar
st.sidebar.markdown("""
**Iniciação Científica Ensino Médio** **Instituição:** UTFPR  
**Acadêmico:** Eduardo Fonseca Silveira  

**Professoras Responsáveis:**
* Fernanda Cavicchioli Zola  
* Daiane Maria De Genaro Chiroli
""")

st.sidebar.info("O sistema normaliza automaticamente os pesos caso a soma seja diferente de 1.")

# --- ENTRADA DE DADOS ---
with st.expander("1. Definição do Projeto", expanded=True):
    titulo_projeto = st.text_input("Título da Análise", "Análise de Desempenho")
    c1, c2 = st.columns(2)
    qtd_criterios = c1.number_input("Qtd de Critérios", 1, 15, 3)
    qtd_alternativas = c2.number_input("Qtd de Alternativas", 2, 50, 3)

st.header("2. Critérios e Pesos")
nomes_crit, pesos_brutos, tipos = [], [], []
cols = st.columns(qtd_criterios)
for i in range(qtd_criterios):
    with cols[i]:
        nome = st.text_input(f"Nome C{i+1}", f"C{i+1}", key=f"n_{i}")
        peso = st.number_input(f"Peso C{i+1}", 0.0, 10.0, 1.0, key=f"p_{i}")
        tipo = st.selectbox(f"Tipo C{i+1}", ["lucro", "custo"], key=f"t_{i}")
        nomes_crit.append(nome); pesos_brutos.append(peso); tipos.append(tipo)

# Normalização de Pesos
soma_p = sum(pesos_brutos)
pesos_norm = [p/soma_p for p in pesos_brutos] if soma_p > 0 else [1/qtd_criterios]*qtd_criterios

st.header("3. Dados das Alternativas")
dados = []
for j in range(qtd_alternativas):
    row = st.columns([2] + [1] * qtd_criterios)
    nome_alt = row[0].text_input(f"Opção {j+1}", f"Opção {j+1}", key=f"alt_{j}")
    val_linha = [nome_alt]
    for i in range(qtd_criterios):
        v = row[i+1].number_input(f"{nomes_crit[i]}", value=10, step=1, key=f"v_{j}_{i}")
        val_linha.append(v)
    dados.append(val_linha)

df_base = pd.DataFrame(dados, columns=["Alternativa"] + nomes_crit)

# --- ENGINE TOPSIS ---
def executar_topsis(df, w, t):
    matrix = df.drop(columns=['Alternativa']).values.astype(float)
    norm = matrix / np.sqrt((matrix**2).sum(axis=0) + 1e-9)
    weighted = norm * w
    v_pos, v_neg = [], []
    for i in range(len(t)):
        if t[i] == 'lucro':
            v_pos.append(np.max(weighted[:, i])); v_neg.append(np.min(weighted[:, i]))
        else:
            v_pos.append(np.min(weighted[:, i])); v_neg.append(np.max(weighted[:, i]))
    s_pos = np.sqrt(((weighted - v_pos)**2).sum(axis=1))
    s_neg = np.sqrt(((weighted - v_neg)**2).sum(axis=1))
    return s_neg / (s_pos + s_neg + 1e-9)

# --- PROCESSAMENTO E BOTÕES ---
st.markdown("---")
if st.button("📊 PROCESSAR RANKING FINAL"):
    scores = executar_topsis(df_base, pesos_norm, tipos)
    df_base['Score TOPSIS'] = scores
    ranking = df_base.sort_values(by='Score TOPSIS', ascending=False).reset_index(drop=True)
    ranking.index += 1
    
    st.balloons()
    st.success(f"Melhor alternativa: **{ranking.iloc[0]['Alternativa']}**")
    
    st.dataframe(ranking, use_container_width=True)
    st.bar_chart(ranking.set_index('Alternativa')['Score TOPSIS'])

    # --- BOTÕES DE DOWNLOAD LADO A LADO ---
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        csv = ranking.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV", csv, "resultado.csv", "text/csv")
        
    with col_btn2:
        # Gerando os bytes do PDF com as informações atualizadas
        pdf_output = gerar_pdf(ranking, titulo_projeto, "Eduardo Fonseca Silveira")
        st.download_button(
            label="📄 Gerar Relatório PDF",
            data=pdf_output,
            file_name="relatorio_topsis.pdf",
            mime="application/pdf"
        )

st.caption("Eduardo Fonseca Silveira | UTFPR 2026")

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB0Ee_p0c1BDFcYZoH4C_v46kH7cLBJSDs",
  authDomain: "apppy-9675a.firebaseapp.com",
  projectId: "apppy-9675a",
  storageBucket: "apppy-9675a.firebasestorage.app",
  messagingSenderId: "1022579734978",
  appId: "1:1022579734978:web:d7a7ac0e0e84ac7fd5be5b",
  measurementId: "G-4P42W63H32"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
