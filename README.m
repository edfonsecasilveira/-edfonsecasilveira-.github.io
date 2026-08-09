import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import datetime
from openai import OpenAI

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

# --- FUNÇÃO PARA GERAR PDF COMPLETO ---
def gerar_pdf(ranking, matrizes, titulo, analise_ia=""):
    pdf = FPDF()
    pdf.add_page()
    
    # Cabeçalho
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, "Relatorio Detalhado TOPSIS", ln=True, align='C')
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(200, 10, "UTFPR - Iniciacao Cientifica Ensino Medio", ln=True, align='C')
    pdf.ln(5)

    # Info Projeto
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 7, f"Analise: {titulo}", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 7, f"Academico: Eduardo Fonseca Silveira | Data: {datetime.datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(5)

    # Ranking Final
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "1. Ranking Final", ln=True)
    pdf.set_font("Helvetica", '', 9)
    pdf.cell(20, 8, "Rank", 1); pdf.cell(100, 8, "Alternativa", 1); pdf.cell(40, 8, "Score", 1); pdf.ln()
    for i, row in ranking.iterrows():
        pdf.cell(20, 8, str(i), 1)
        pdf.cell(100, 8, str(row['Alternativa']), 1)
        pdf.cell(40, 8, f"{row['Score TOPSIS']:.4f}", 1); pdf.ln()
    
    # Incluir Análise da IA no PDF se existir
    if analise_ia:
        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, "2. Relatorio Interpretativo da IA", ln=True)
        pdf.set_font("Helvetica", '', 10)
        texto_limpo = analise_ia.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 5, texto_limpo)
    
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "3. Matrizes Intermediarias (Resumo)", ln=True)
    pdf.set_font("Helvetica", '', 8)
    pdf.multi_cell(0, 5, "As matrizes de decisao normalizada, ponderada e as distancias euclidianas foram processadas conforme o rigor matematico do metodo TOPSIS.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- FUNÇÃO DA IA ---
def gerar_analise_ia(api_key, ranking_df, pesos, criterios, tipos):
    try:
        client = OpenAI(api_key=api_key)
        
        resumo_problema = f"Critérios avaliados: {', '.join(criterios)}\n"
        resumo_problema += f"Pesos de cada critério: {pesos}\n"
        resumo_problema += f"Tipos (lucro/custo): {tipos}\n\n"
        resumo_problema += "Tabela de Classificação Final (Ranking):\n"
        resumo_problema += ranking_df.to_string()

        prompt = f"""
        Você é um especialista em Análise de Decisão Multicritério (MCDA). 
        Analise o seguinte resultado do método TOPSIS e forneça um relatório interpretativo curto, direto e acadêmico para o usuário.
        
        {resumo_problema}
        
        Escreva uma resposta estruturada contendo:
        1. Justificativa do porquê a alternativa vencedora ficou em primeiro lugar com base nos pesos descritos.
        2. Destaque rápido de qual critério foi o "divisor de águas" para essa escolha.
        3. Um parágrafo de conclusão ou recomendação prática.
        
        Importante: Responda em português, use formatação Markdown simples e evite respostas muito longas. Não use emojis complexos ou caracteres especiais fora do padrão latino para não quebrar o PDF.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente científico especialista em tomada de decisão matemática."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices.message.content
    except Exception as e:
        return f"Erro ao conectar com a IA: {str(e)}"

# --- SIDEBAR ---
st.sidebar.image("https://portal.utfpr.edu.br/icones/cabecalho/logo-utfpr/@@images/image.png", width=180)
st.sidebar.markdown("---")

# Configuração da API Key na Sidebar
st.sidebar.subheader("🤖 Configuração da IA")
api_key_input = st.sidebar.text_input("Insira sua OpenAI API Key", type="password", help="Pegue sua chave no painel da OpenAI.")

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
st.info("💡 Agora você pode usar pontos para números decimais (ex: 1.50 ou 0.75).")
nomes_crit, pesos_brutos, tipos = [], [], []
cols = st.columns(qtd_c)
for i in range(qtd_c):
    with cols[i]:
        n = st.text_input(f"C{i+1}", f"Critério {i+1}", key=f"nc_{i}")
        p = st.number_input(f"Peso C{i+1}", 0.0, 100.0, 1.0, step=0.01, format="%.2f", key=f"p_{i}")
        t = st.selectbox(f"Objetivo C{i+1}", ["lucro", "custo"], key=f"t_{i}")
        nomes_crit.append(n); pesos_brutos.append(p); tipos.append(t)

soma_p = sum(pesos_brutos)
pesos_norm = [p/soma_p for p in pesos_brutos] if soma_p > 0 else [1/qtd_c]*qtd_c

st.header("3. Matriz de Decisão")
dados = []
for j in range(qtd_a):
    row = st.columns([2] + [1] * qtd_c)
    nome_alt = row[0].text_input(f"Alternativa {j+1}", f"Opção {j+1}", key=f"alt_{j}")
    val_linha = [nome_alt]
    for i in range(qtd_c):
        v = row[i+1].number_input(f"{nomes_crit[i]}", value=0.0, step=0.01, format="%.2f", key=f"v_{j}_{i}")
        val_linha.append(v)
    dados.append(val_linha)

df_base = pd.DataFrame(dados, columns=["Alternativa"] + nomes_crit)

# --- ENGINE TOPSIS ---
def executar_topsis_completo(df, w, t):
    matrix = df.drop(columns=['Alternativa']).values.astype(float)
    norm = matrix / np.sqrt((matrix**2).sum(axis=0) + 1e-9)
    weighted = norm * w
    
    v_pos, v_neg = [], []
    for i in range(len(t)):
        if t[i] == 'lucro':
            v_pos.append(np.max(weighted[:, i]))
            v_neg.append(np.min(weighted[:, i]))
        else:
            v_pos.append(np.min(weighted[:, i]))
            v_neg.append(np.max(weighted[:, i]))
    
    s_pos = np.sqrt(((weighted - v_pos)**2).sum(axis=1))
    s_neg = np.sqrt(((weighted - v_neg)**2).sum(axis=1))
    scores = s_neg / (s_pos + s_neg + 1e-9)
    
    intermediarias = {
        "Normalizada": pd.DataFrame(norm, columns=nomes_crit),
        "Ponderada": pd.DataFrame(weighted, columns=nomes_crit),
        "Ideais": pd.DataFrame([v_pos, v_neg], columns=nomes_crit, index=["Ideal (+)", "Anti-Ideal (-)"]),
        "Distâncias": pd.DataFrame({"S+ (Ideal)": s_pos, "S- (Anti-Ideal)": s_neg})
    }
    return scores, intermediarias

# --- PROCESSAMENTO ---
st.markdown("---")
if st.button("📊 EXECUTAR ANÁLISE COMPLETA"):
    scores, matrizes = executar_topsis_completo(df_base, pesos_norm, tipos)
    
    df_base['Score TOPSIS'] = scores
    ranking = df_base.sort_values(by='Score TOPSIS', ascending=False).reset_index(drop=True)
    ranking.index += 1
    
    st.session_state['resultado'] = ranking
    st.session_state['matrizes'] = matrizes
    
    st.balloons()
    st.success(f"🏆 Melhor Escolha: **{ranking.iloc[0]['Alternativa']}**")
    
    # Mostrar Ranking
    st.subheader("🏁 Ranking Final")
    st.dataframe(ranking.style.format({"Score TOPSIS": "{:.4f}"}), use_container_width=True)

    # --- RELATÓRIO DA IA ---
    if api_key_input:
        with st.spinner("🤖 IA analisando os resultados matemáticos..."):
            analise = gerar_analise_ia(api_key_input, ranking, pesos_brutos, nomes_crit, tipos)
            st.session_state['analise_ia'] = analise
            
        st.subheader("🤖 Relatório Interpretativo da IA")
        st.markdown(analise)
    else:
        st.session_state['analise_ia'] = ""
        st.info("ℹ️ Adicione sua OpenAI API Key na barra lateral para gerar um relatório interpretativo automático via IA.")

    # Exibir Matrizes Intermediárias
    with st.expander("📂 Ver Memória de Cálculo (Matrizes Intermediárias)"):
        st.write("**Matriz Normalizada**")
        st.dataframe(matrizes["Normalizada"])
        st.write("**Matriz Ponderada (Pesos Aplicados)**")
        st.dataframe(matrizes["Ponderada"])
        st.write("**Soluções Ideais**")
        st.dataframe(matrizes["Ideais"])
        st.write("**Distâncias Euclidianas**")
        st.dataframe(matrizes["Distâncias"])

# --- DOWNLOADS ---
if 'resultado' in st.session_state:
    c_down1, c_down2 = st.columns(2)
    with c_down1:
csv = st.session_state['resultado'].to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar CSV do Ranking", csv, "ranking.csv", "text/csv") 
com c_down2:
texto_ia = st.session_state.get('analise_ia', "")
pdf_out = gerar_pdf(st.session_state['resultado'], st.session_state['matrizes'],
titulo_projeto, analise_ia=texto_ia)
st.download_button("📄 Baixar Relatório Completo (PDF)", pdf_out, "relatorio_topsis.pdf",
"application/pdf")

st.markdown("---")
st.caption("Eduardo Fonseca Silveira | UTFPR 2026")
