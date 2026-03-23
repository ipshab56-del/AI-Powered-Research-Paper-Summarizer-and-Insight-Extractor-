# requirements


# neo4j
# pyvis
# openpyxl

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="AI Paper Insight",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gemini_file import ask_gemini

from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import io

# page title 

# Title
# color:#00FFFF;
st.markdown("""
<style>
.main-header {
    font-family: 'Segoe UI', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #0077aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: glow 2s ease-in-out infinite alternate;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}
.main-subtitle {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.4rem;
    color: #a0a0a0;
    text-align: center;
    margin-bottom: 2rem;
}
@keyframes glow {
    from { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3)); }
    to { filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.6)); }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 AI-Powered Research Paper Insight Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Unleash AI magic on your research papers • Ask • Discover • Visualize</div>', unsafe_allow_html=True)

# Tabs

tab1, tab2 = st.tabs(['Research Paper QA', "Knowledge Graph Explorer"])


# Tab 1-> RAG Research Paper Question Answering 

with tab1:
    st.markdown("""
    <style>
    /* Pro + Playful Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .css-1d391kg {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    .stTextInput > label > div > div > input {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(0,212,255,0.1));
        border: 2px solid transparent;
        border-radius: 15px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,212,255,0.1);
    }
    .stTextInput > label > div > div > input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 3px rgba(0,212,255,0.2), 0 0 30px rgba(0,212,255,0.4);
        transform: scale(1.02);
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 10px 30px rgba(0,212,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,212,255,0.6);
        background: linear-gradient(135deg, #0099cc, #00d4ff);
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(0,212,255,0.1));
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(0,212,255,0.3);
        backdrop-filter: blur(15px);
    }
    .stExpander > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00d4ff;
    }
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0,212,255,0.3);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(0,212,255,0.1)) !important;
        border-radius: 15px 15px 0 0 !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        color: white !important;
        font-weight: 600;
        padding: 1rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(0,153,204,0.3)) !important;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"][aria-selected=true] {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: white !important;
        box-shadow: 0 5px 20px rgba(0,212,255,0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_vector_db():

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "research_papers_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_db


    vector_db = load_vector_db()

    # Show total papers
    col_stats, col_space = st.columns([1,3])
    with col_stats:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color: #00d4ff; text-align: center;'>📚</h2>
            <h3 style='color: white; text-align: center;'>"""+str(vector_db.index.ntotal)+""" Papers</h3>
        </div>
        """, unsafe_allow_html=True)

    # User input
    user_query = st.text_input("🔎 Ask a question about research papers:")

    # Search button
    if st.button("Search"):

        results = vector_db.similarity_search(user_query, k=3)

        content = ""

        for idx, doc in enumerate(results, 1):

            title = doc.metadata.get("title", f"Paper {idx}")

            content += f"""
            Paper Title: {title}

            Paper Content:
            {doc.page_content}

            """

        print("whole content",content )
        # Call Gemini
        with st.spinner("🤖 Analyzing research papers and generating insights..."):
            response = ask_gemini(content, user_query)
        print("gemini response", response)
        # -------------------------
        # Extract Answer and Paper
        # -------------------------

        answer = ""
        paper_titles = []

        if "Research Paper:" in response:
            parts = response.split("Research Paper:")
            
            answer = parts[0].replace("Answer:", "").strip()

            papers_text = parts[1].strip()

            # Split multiple papers by comma
            paper_titles = [p.strip() for p in papers_text.split(",")]

        else:
            answer = response.strip()

        # Show AI answer
        st.markdown("""
        <div class='css-1d391kg'>
            <h3 style='color: #00d4ff;'>🤖 AI Generated Insight</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>"""+answer+"""</p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # Show Only Relevant Paper
        # -------------------------

        if paper_titles and "none" not in [p.lower() for p in paper_titles]:

            col1, col2 = st.columns(2)
            relevant_found = False
            for i, doc in enumerate(results):
                title = doc.metadata.get("title", f"Paper {i+1}")
                if any(p.lower() in title.lower() for p in paper_titles):
                    with col1 if not relevant_found else col2:
                        st.markdown(f"""
                        <div class='css-1d391kg'>
                            <h4 style='color: #00d4ff;'>📄 {title}</h4>
                            <p>{doc.page_content[:500]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    relevant_found = True

        else:
            st.warning("No relevant research paper found.")
            
            
            
with tab2:
    st.subheader("Knowledge Graph Explorer")            
    
    
    # Neo4j connection
    @st.cache_resource
    def get_driver():
        return GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=('neo4j','neo4j123')
        )
    
    driver = get_driver()
    
    # get domain
    
    @st.cache_data
    def get_domain():
        try:
            query=""" 
            MATCH (d:Domain)
            RETURN d.name as domain
            """
            
            with driver.session() as session:
                result = session.run(query)
                domains = [r["domain"] for r in result]
            
            normalized={}
            
            for d in domains:
                normalized[d.lower()]=d.title()
                
            return sorted(normalized.values())
        except Exception as e:
            st.error(f"Neo4j connection failed (server not running?): {str(e)[:200]}")
            st.info("Start Neo4j at neo4j://127.0.0.1:7687 (user: neo4j, pass: neo4j123)")
            return []
        
        
    # domain selector
    domain = st.selectbox(
        "Select Research Domain",
        get_domain()
    )
    
    # Fetch Graph Data
    def get_graph_data(domain):
        try:
            query= """ 
            
            MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
            WHERE toLower(d.name) = toLower($domain)
            
            OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
            OPTIONAL MATCH (p)-[:USES]-(m:Method)
            
            RETURN p.title AS paper,
            a.name AS author,
            m.name AS method,
            d.name AS domain
            """
            
            with driver.session() as session:
                result = session.run(query, domain=domain)
                return [r.data() for r in result]
        except Exception as e:
            st.error(f"Neo4j query failed: {str(e)[:200]}")
            return []
        
    
    # Draw Graph
    
    def draw_graph(data):
        
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#0a0a0a",
            font_color="white",
            physics=True
        )
        
        for row in data:
            paper = row['paper']
            author = row['author']
            method = row['method']
            domain = row['domain']
            
            net.add_node(paper, label=paper, color="orange")
            
            if author:
                net.add_node(author, label=author, color="skyblue")
                net.add_edge(author, paper)
                
            if method:
                net.add_node(method, label=method, color="green")
                net.add_edge(paper, method)
                
            if domain:
                net.add_node(domain, label=domain, color="purple")
                net.add_edge(paper, domain)
                
        net.save_graph("graph.html")
        
        with open("graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600)
    
    
    # display data
    if domain:
      st.subheader(f"Knowledge Graph for Domain: {domain}")  
      
      data = get_graph_data(domain)
      
      if len(data)==0:
          st.warning("No papers found")
          
      else:
        df= pd.DataFrame(data)
          
        papers = df['paper'].nunique()
        authors = df['author'].nunique()
        methods = df['method'].nunique()
          
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #00d4ff;'>📚</h2>
                <h3 style='color: white;'>{papers}</h3>
                <p style='color: #a0a0a0;'>Papers</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #4ade80;'>👥</h2>
                <h3 style='color: white;'>{authors}</h3>
                <p style='color: #a0a0a0;'>Authors</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #f59e0b;'>⚙️</h2>
                <h3 style='color: white;'>{methods}</h3>
                <p style='color: #a0a0a0;'>Methods</p>
            </div>
            """, unsafe_allow_html=True)
          
        st.divider()
          
          
        st.markdown('<h3 style="color: #00d4ff;">📊 Filtered Research Data</h3>')
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "paper": st.column_config.TextColumn("Paper", help="Paper Title"),
                "author": st.column_config.TextColumn("Author", help="Authors"),
                "method": st.column_config.TextColumn("Method", help="Methods Used"),
            }
        )
          
          
        #Export Excel Button
        
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        st.download_button(
            "Export Excel",
            excel_buffer,
            file_name=f"{domain}_research_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
          
        st.divider()
        
        st.subheader("Knowledge Graph Visualization")
        draw_graph(data)
          
          
        
            