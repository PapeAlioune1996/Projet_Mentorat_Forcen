import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io  
import warnings

# Désactiver les avertissements liés à la conversion PyArrow
warnings.filterwarnings('ignore', category=UserWarning, message='.*Arrow.*')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Réadmissions Hospitalières",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Personnalisation du style de la sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #008080;
        }
        [data-testid="stSidebar"] .sidebar-content {
            background-color: #008080;
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.title('📊 Dashboard Analyse des Réadmissions Hospitalières')
st.markdown("""
Ce dashboard permet d'explorer les données de réadmission hospitalière sur 10 ans pour identifier 
les facteurs clés influençant les réadmissions hospitalières.
""")

# Chargement des données
@st.cache_data
def load_data():
    # Lire le fichier CSV
    df = pd.read_csv("hospital_readmissions.csv")
    
    # Afficher les valeurs uniques pour le débogage
    #st.write("Valeurs uniques de glucose_test:", df['glucose_test'].unique())
    #st.write("Valeurs uniques de A1Ctest:", df['A1Ctest'].unique())
    
    # Convertir explicitement les colonnes catégorielles en string pour éviter les problèmes avec PyArrow
    categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
                     'glucose_test', 'A1Ctest', 'change', 'diabetes_med', 'readmitted']
    
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    return df

# Fonction pour afficher un message de chargement pendant le chargement des données
with st.spinner('Chargement des données en cours...'):
    df = load_data()

# Barre latérale
st.sidebar.title("Options d'Affichage")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Sélectionnez une section:",
    ["Aperçu des Données", "Démographie", "Variables Cliniques", 
     "Variables Diabète", "Historique Médical", "Corrélations"]
)

#=============================================================================
# PAGE 1: APERÇU DES DONNÉES
#=============================================================================
if page == "Aperçu des Données":
    # Style moderne pour toute la page
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
        }
        .section-header {
            font-size: 1.8em;
            color: #2c3e50;
            margin: 1em 0;
            padding: 0.5em;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            color: #1f77b4;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .data-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="main-header">📊 Aperçu des Données</h1>', unsafe_allow_html=True)

    # Section 1: Métriques Clés
    st.markdown('<h2 class="section-header">📈 Métriques Clés</h2>', unsafe_allow_html=True)
    
    # Créer 4 colonnes pour les métriques principales
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Total Patients</div>
            </div>
        """, unsafe_allow_html=True)
        st.metric(
            label="",
            value=f"{df.shape[0]:,}",
            delta=None
        )

    with metric_col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Patients Réadmis</div>
            </div>
        """, unsafe_allow_html=True)
        readmitted = df[df['readmitted'] == 'yes'].shape[0]
        st.metric(
            label="",
            value=f"{readmitted:,}",
            delta=f"{readmitted/df.shape[0]*100:.1f}%"
        )

    with metric_col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Patients Non Réadmis</div>
            </div>
        """, unsafe_allow_html=True)
        not_readmitted = df[df['readmitted'] == 'no'].shape[0]
        st.metric(
            label="",
            value=f"{not_readmitted:,}",
            delta=f"{not_readmitted/df.shape[0]*100:.1f}%"
        )

    with metric_col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Taux de Réadmission</div>
            </div>
        """, unsafe_allow_html=True)
        readmission_rate = (readmitted/df.shape[0]*100)
        st.metric(
            label="",
            value=f"{readmission_rate:.1f}%",
            delta=None
        )

    # Section 2: Visualisations
    st.markdown('<h2 class="section-header">📊 Visualisations</h2>', unsafe_allow_html=True)
    
    # Créer 2 colonnes pour les graphiques
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Distribution de la Réadmission")
        readmission_counts = df['readmitted'].value_counts().reset_index()
        readmission_counts.columns = ['Réadmission', 'Nombre']
        
        fig = px.pie(
            readmission_counts, 
            values='Nombre', 
            names='Réadmission', 
            title='Distribution des Réadmissions',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Distribution des Variables Numériques")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_var = st.selectbox("Sélectionnez une variable", numeric_cols)
        
        fig = px.histogram(
            df,
            x=selected_var,
            color='readmitted',
            title=f'Distribution de {selected_var}',
            color_discrete_map={'yes': 'firebrick', 'no': 'royalblue'},
            opacity=0.7
        )
        fig.update_layout(
            title_x=0.5,
            showlegend=True,
            legend_title="Réadmission"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Données Brutes
    st.markdown('<h2 class="section-header">📋 Données Brutes</h2>', unsafe_allow_html=True)
    
    # Créer 2 colonnes pour les tableaux
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Échantillon du Dataset")
        st.dataframe(
            df.head().style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with data_col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Statistiques Descriptives")
        st.dataframe(
            df.describe().style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Informations sur le Dataset
    st.markdown('<h2 class="section-header">ℹ️ Informations sur le Dataset</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.markdown('</div>', unsafe_allow_html=True)

#=============================================================================
# PAGE 2: DÉMOGRAPHIE
#=============================================================================
elif page == "Démographie":
    # Style moderne pour la page démographie
    st.markdown("""
        <style>
        .demography-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
            background: linear-gradient(45deg, #1f77b4, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .demography-section {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .demography-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-box {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="demography-header">👴👩 Analyse Démographique</h1>', unsafe_allow_html=True)

    # Section 1: Vue d'ensemble
    st.markdown('<div class="demography-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="demography-title">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    # Créer 3 colonnes pour les métriques clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        age_groups = df['age'].nunique()
        st.metric("Nombre de Groupes d'Âge", f"{age_groups}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        most_common_age = df['age'].mode()[0]
        total_patients_group_le_plus_represente = df['age'].value_counts().max()
        percentage_most_common = (total_patients_group_le_plus_represente / len(df)) * 100
        st.metric(
            "Groupe d'Âge le Plus Représenté", 
            f"{most_common_age}",
            f"Total: {total_patients_group_le_plus_represente:,} ({percentage_most_common:.1f}%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        least_common_age = df['age'].value_counts().index[-1]
        total_patients_group_le_moins_represente = df['age'].value_counts().min()
        percentage_least_common = (total_patients_group_le_moins_represente / len(df)) * 100
        st.metric(
            "Groupe d'Âge le Moins Représenté", 
            f"{least_common_age}",
            f"Total: {total_patients_group_le_moins_represente:,} ({percentage_least_common:.1f}%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        # Groupe d'âge avec le plus de réadmissions
        age_readmissions = df[df['readmitted'] == 'yes']['age'].value_counts()
        most_readmitted_age = age_readmissions.index[0]
        total_most_readmitted = age_readmissions.max()
        percentage_most_readmitted = (total_most_readmitted / len(df[df['readmitted'] == 'yes'])) * 100
        st.metric(
            "Groupe d'Âge avec le Plus de Réadmissions", 
            f"{most_readmitted_age}",
            f"Total: {total_most_readmitted:,} ({percentage_most_readmitted:.1f}%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Groupe d'âge avec le moins de réadmissions
        least_readmitted_age = age_readmissions.index[-1]
        total_least_readmitted = age_readmissions.min()
        percentage_least_readmitted = (total_least_readmitted / len(df[df['readmitted'] == 'yes'])) * 100
        st.metric(
            "Groupe d'Âge avec le Moins de Réadmissions", 
            f"{least_readmitted_age}",
            f"Total: {total_least_readmitted:,} ({percentage_least_readmitted:.1f}%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Distribution par Âge
    st.markdown('<div class="demography-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="demography-title">📈 Distribution par Âge</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des patients par âge
        age_counts = df['age'].value_counts().reset_index()
        age_counts.columns = ['Groupe d\'âge', 'Nombre']
        
        fig = px.bar(
            age_counts, 
            x='Groupe d\'âge', 
            y='Nombre', 
            title='Distribution des Patients par Groupe d\'Âge',
            color='Nombre',
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(
            xaxis_title="Groupe d'âge",
            yaxis_title="Nombre de patients",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Taux de réadmission par âge
        age_readmit = pd.crosstab(df['age'], df['readmitted'], normalize='index') * 100
        age_readmit = age_readmit.reset_index()
        
        fig = px.bar(
            age_readmit, 
            x='age', 
            y='yes', 
            title='Taux de Réadmission par Groupe d\'Âge',
            labels={'age': 'Groupe d\'âge', 'yes': 'Taux de réadmission (%)'},
            color='yes',
            color_continuous_scale='Reds',
            text_auto=True
        )
        fig.update_layout(
            xaxis_title="Groupe d'âge",
            yaxis_title="Taux de réadmission (%)",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Analyse Détaillée
    st.markdown('<div class="demography-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="demography-title">🔍 Analyse Détaillée</h2>', unsafe_allow_html=True)
    
    # Créer les données pour la pyramide
    age_gender_data = pd.DataFrame({
        'Âge': df['age'].unique(),
        'Total': df['age'].value_counts(),
        'Réadmis': df[df['readmitted'] == 'yes']['age'].value_counts()
    }).fillna(0)
    
    # Calculer les pourcentages
    age_gender_data['% Réadmission'] = (age_gender_data['Réadmis'] / age_gender_data['Total']) * 100
    
    # Créer le graphique pyramide
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=age_gender_data['Âge'],
        x=age_gender_data['% Réadmission'],
        name='Taux de Réadmission',
        orientation='h',
        marker=dict(
            color='firebrick',
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        )
    ))
    
    fig.update_layout(
        title='Pyramide des Taux de Réadmission par Âge',
        xaxis_title='Taux de Réadmission (%)',
        yaxis_title='Groupe d\'âge',
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Facteurs de Risque par Âge
    st.markdown('<div class="demography-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="demography-title">⚠️ Facteurs de Risque par Âge</h2>', unsafe_allow_html=True)
    
    # Créer un DataFrame pour la carte de chaleur
    risk_by_age = pd.DataFrame()
    
    for age in df['age'].unique():
        age_data = df[df['age'] == age]
        risk_by_age.loc[age, 'Durée Hospitalisation'] = age_data['time_in_hospital'].mean()
        risk_by_age.loc[age, 'Nombre Médicaments'] = age_data['n_medications'].mean()
        risk_by_age.loc[age, 'Nombre Procédures'] = age_data['n_procedures'].mean()
        risk_by_age.loc[age, 'Visites Urgences'] = age_data['n_emergency'].mean()
    
    # Créer la carte de chaleur
    fig = px.imshow(
        risk_by_age,
        text_auto=True,
        color_continuous_scale='Reds',
        aspect="auto",
        title='Facteurs de Risque par Groupe d\'Âge'
    )
    
    fig.update_layout(
        xaxis_title='Facteurs de Risque',
        yaxis_title='Groupe d\'âge',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: Distribution des Facteurs de Risque
    st.markdown('<div class="demography-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="demography-title">📊 Distribution des Facteurs de Risque</h2>', unsafe_allow_html=True)
    
    # Sélectionner les facteurs à afficher
    risk_factors = ['time_in_hospital', 'n_medications', 'n_procedures', 'n_emergency']
    
    # Créer le graphique
    fig = make_subplots(rows=2, cols=2, subplot_titles=risk_factors)
    
    for i, factor in enumerate(risk_factors):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Box(
                y=df[factor],
                x=df['age'],
                name=factor,
                boxpoints='outliers'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="Distribution des Facteurs de Risque par Groupe d'Âge",
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

#=============================================================================
# PAGE 3: VARIABLES CLINIQUES
#=============================================================================
elif page == "Variables Cliniques":
    # Style moderne pour la page variables cliniques
    st.markdown("""
        <style>
        .clinical-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
            background: linear-gradient(45deg, #1f77b4, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .clinical-section {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .clinical-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-box {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        .variable-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="clinical-header">🏥 Analyse des Variables Cliniques</h1>', unsafe_allow_html=True)

    # Section 1: Vue d'ensemble
    st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="clinical-title">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    # Créer 4 colonnes pour les métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_time = df['time_in_hospital'].mean()
        st.metric("Durée Moyenne d'Hospitalisation", f"{avg_time:.1f} jours")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_meds = df['n_medications'].mean()
        st.metric("Nombre Moyen de Médicaments", f"{avg_meds:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_procedures = df['n_procedures'].mean()
        st.metric("Nombre Moyen de Procédures", f"{avg_procedures:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_lab = df['n_lab_procedures'].mean()
        st.metric("Nombre Moyen d'Examens Labo", f"{avg_lab:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Sélection et Analyse de Variable
    st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="clinical-title">🔍 Analyse Détaillée</h2>', unsafe_allow_html=True)
    
    # Sélecteur de variable avec style moderne
    st.markdown('<div class="variable-selector">', unsafe_allow_html=True)
    clinical_var = st.selectbox(
        "Sélectionnez une variable clinique à analyser:",
        ["time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications", "diag_1"],
        format_func=lambda x: {
            "time_in_hospital": "Durée d'Hospitalisation",
            "n_lab_procedures": "Nombre d'Examens Laboratoire",
            "n_procedures": "Nombre de Procédures",
            "n_medications": "Nombre de Médicaments",
            "diag_1": "Diagnostic Principal"
        }[x]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if clinical_var == "diag_1":
        # Analyse du diagnostic principal
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des diagnostics
            diag_counts = df['diag_1'].value_counts().reset_index()
            diag_counts.columns = ['Diagnostic', 'Nombre']
            
            fig = px.bar(
                diag_counts, 
                x='Diagnostic', 
                y='Nombre', 
                title='Distribution des Diagnostics Principaux',
                color='Nombre',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig.update_layout(
                xaxis_title="Diagnostic principal",
                yaxis_title="Nombre de patients",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Taux de réadmission par diagnostic
            diag_readmit = pd.crosstab(df['diag_1'], df['readmitted'], normalize='index') * 100
            diag_readmit = diag_readmit.reset_index()
            diag_readmit = diag_readmit.sort_values(by='yes', ascending=False)
            
            fig = px.bar(
                diag_readmit, 
                x='diag_1', 
                y='yes', 
                title='Taux de Réadmission par Diagnostic Principal',
                labels={'diag_1': 'Diagnostic principal', 'yes': 'Taux de réadmission (%)'},
                color='yes',
                color_continuous_scale='Reds',
                text_auto=True
            )
            fig.update_layout(
                xaxis_title="Diagnostic principal",
                yaxis_title="Taux de réadmission (%)",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Analyse des variables numériques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution de la variable
            fig = px.histogram(
                df, 
                x=clinical_var, 
                nbins=30,
                title=f'Distribution de {clinical_var}',
                color_discrete_sequence=['royalblue'],
                opacity=0.7
            )
            fig.update_layout(
                xaxis_title=clinical_var,
                yaxis_title="Nombre de patients",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot par statut de réadmission
            fig = px.box(
                df, 
                x='readmitted', 
                y=clinical_var, 
                title=f'{clinical_var} par Statut de Réadmission',
                color='readmitted',
                color_discrete_map={'yes': 'firebrick', 'no': 'royalblue'},
                points="all"
            )
            fig.update_layout(
                xaxis_title="Réadmission",
                yaxis_title=clinical_var,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives avec style moderne
        st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="clinical-title">📊 Statistiques Descriptives</h3>', unsafe_allow_html=True)
        
        stats = df.groupby('readmitted')[clinical_var].describe().reset_index()
        stats_style = stats.style.background_gradient(cmap='Blues')
        st.dataframe(stats_style, use_container_width=True)
        
        # Graphique plus détaillé pour time_in_hospital
        if clinical_var == "time_in_hospital":
            st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="clinical-title">⏱️ Analyse de la Durée d\'Hospitalisation</h3>', unsafe_allow_html=True)
            
            # Créer des groupes pour le temps d'hospitalisation
            df['time_group'] = pd.cut(df['time_in_hospital'], 
                                     bins=[0, 3, 7, 14],
                                     labels=['1-3 jours', '4-7 jours', '8-14 jours'])
            
            time_readmit = pd.crosstab(df['time_group'], df['readmitted'], normalize='index') * 100
            time_readmit = time_readmit.reset_index()
            
            fig = px.bar(
                time_readmit, 
                x='time_group', 
                y='yes', 
                title='Taux de Réadmission par Durée d\'Hospitalisation',
                labels={'time_group': 'Durée d\'hospitalisation', 'yes': 'Taux de réadmission (%)'},
                color='yes',
                color_continuous_scale='Reds',
                text_auto=True
            )
            fig.update_layout(
                xaxis_title="Durée d'hospitalisation",
                yaxis_title="Taux de réadmission (%)",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Corrélations avec les Variables Cliniques
    st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="clinical-title">🔗 Corrélations</h2>', unsafe_allow_html=True)
    
    # Sélectionner les variables numériques
    numeric_vars = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications']
    
    # Créer la matrice de corrélation
    corr_matrix = df[numeric_vars].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="auto",
        title='Corrélation entre Variables Cliniques'
    )
    fig.update_layout(
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

#=============================================================================
# PAGE 4: VARIABLES DIABÈTE
#=============================================================================
elif page == "Variables Diabète":
    # Style moderne pour la page variables diabète
    st.markdown("""
        <style>
        .diabetes-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
            background: linear-gradient(45deg, #1f77b4, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .diabetes-section {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .diabetes-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-box {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        .variable-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="diabetes-header">🩸 Analyse des Variables liées au Diabète</h1>', unsafe_allow_html=True)

    # Section 1: Vue d'ensemble
    st.markdown('<div class="diabetes-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="diabetes-title">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    # Créer 4 colonnes pour les métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        glucose_test_yes = df[df['glucose_test'].isin(['normal', 'high'])].shape[0]
        st.metric(
            "Tests de Glucose Effectués", 
            f"{glucose_test_yes:,}", 
            f"{glucose_test_yes/df.shape[0]*100:.1f}%"
        )
        glucose_test_no = df[df['glucose_test'] == 'no'].shape[0]
        st.metric(
            "Tests de Glucose Non Effectués", 
            f"{glucose_test_no:,}", 
            f"{glucose_test_no/df.shape[0]*100:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        a1c_test_yes = df[df['A1Ctest'].isin(['normal', 'high'])].shape[0]
        st.metric(
            "Tests A1C Effectués", 
            f"{a1c_test_yes:,}", 
            f"{a1c_test_yes/df.shape[0]*100:.1f}%"
        )
        a1c_test_no = df[df['A1Ctest'] == 'no'].shape[0]
        st.metric(
            "Tests A1C Non Effectués", 
            f"{a1c_test_no:,}", 
            f"{a1c_test_no/df.shape[0]*100:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        diabetes_med_yes = df[df['diabetes_med'] == 'yes'].shape[0]
        st.metric("Patients sous Médicaments", f"{diabetes_med_yes:,}", 
                 f"{diabetes_med_yes/df.shape[0]*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        change_yes = df[df['change'] == 'yes'].shape[0]
        st.metric("Changements de Médicaments", f"{change_yes:,}", 
                 f"{change_yes/df.shape[0]*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Analyse Détaillée
    st.markdown('<div class="diabetes-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="diabetes-title">🔍 Analyse Détaillée</h2>', unsafe_allow_html=True)
    
    # Sélecteur de variable avec style moderne
    st.markdown('<div class="variable-selector">', unsafe_allow_html=True)
    diabetes_vars = ['glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    selected_var = st.selectbox(
        "Sélectionnez une variable liée au diabète à analyser:",
        diabetes_vars,
        format_func=lambda x: {
            'glucose_test': 'Test de Glucose',
            'A1Ctest': 'Test A1C',
            'change': 'Changement de Médicaments',
            'diabetes_med': 'Médicaments Diabète'
        }[x]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de la variable
        var_counts = df[selected_var].value_counts().reset_index()
        var_counts.columns = [selected_var, 'Nombre']
        total_patients = len(df)
        var_counts['Pourcentage'] = (var_counts['Nombre'] / total_patients) * 100
        
        fig = px.pie(
            var_counts, 
            values='Nombre', 
            names=selected_var, 
            title=f'Distribution de {selected_var}',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux de réadmission par valeur de la variable
        readmission_rates = []
        for value in df[selected_var].unique():
            subset = df[df[selected_var] == value]
            total_in_category = len(subset)
            readmitted_count = len(subset[subset['readmitted'] == 'yes'])
            if total_in_category > 0:
                rate = (readmitted_count / total_in_category) * 100
                readmission_rates.append({
                    'Valeur': value,
                    'Total patients': total_in_category,
                    'Patients réadmis': readmitted_count,
                    'Taux de réadmission (%)': rate
                })
        
        readmission_df = pd.DataFrame(readmission_rates)
        
        fig = px.bar(
            readmission_df, 
            x='Valeur', 
            y='Taux de réadmission (%)', 
            title=f'Taux de Réadmission par {selected_var}',
            labels={'Valeur': selected_var},
            color='Taux de réadmission (%)',
            color_continuous_scale='Reds',
            text='Taux de réadmission (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%')
        fig.update_layout(
            xaxis_title=selected_var,
            yaxis_title="Taux de réadmission (%)",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Analyse Combinée
    st.markdown('<div class="diabetes-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="diabetes-title">🔄 Analyse Combinée</h2>', unsafe_allow_html=True)
    
    # Créer une nouvelle variable combinant diabetes_med et change
    df['diabetes_change'] = df['diabetes_med'] + '_' + df['change']
    
    # Calculer les taux de réadmission pour la combinaison
    combined_readmit_data = []
    for combo in df['diabetes_change'].unique():
        subset = df[df['diabetes_change'] == combo]
        total_in_combo = len(subset)
        readmitted_count = len(subset[subset['readmitted'] == 'yes'])
        
        if total_in_combo > 0:
            rate = (readmitted_count / total_in_combo) * 100
            combined_readmit_data.append({
                'Combinaison': combo,
                'Total patients': total_in_combo,
                'Patients réadmis': readmitted_count,
                'Taux de réadmission (%)': rate
            })
    
    combined_readmit_df = pd.DataFrame(combined_readmit_data)
    combined_readmit_df = combined_readmit_df.sort_values(by='Taux de réadmission (%)', ascending=False)
    
    fig = px.bar(
        combined_readmit_df, 
        x='Combinaison', 
        y='Taux de réadmission (%)', 
        title='Taux de Réadmission par Combinaison Médicaments Diabète et Changement',
        labels={'Combinaison': 'Diabète médicaments + Changement'},
        color='Taux de réadmission (%)',
        color_continuous_scale='Reds',
        text='Taux de réadmission (%)',
        hover_data=['Total patients', 'Patients réadmis']
    )
    fig.update_traces(texttemplate='%{text:.1f}%')
    fig.update_layout(
        xaxis_title="Médicaments Diabète + Changement", 
        yaxis_title="Taux de réadmission (%)",
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Tableau Comparatif
    st.markdown('<div class="diabetes-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="diabetes-title">📋 Tableau Comparatif</h2>', unsafe_allow_html=True)
    
    # Créer un tableau pour toutes les variables
    diabetes_table = []
    
    for var in diabetes_vars:
        for val in df[var].unique():
            subset = df[df[var] == val]
            total_in_category = len(subset)
            readmitted_count = len(subset[subset['readmitted'] == 'yes'])
            
            if total_in_category > 0:
                row = {
                    'Variable': var,
                    'Valeur': val,
                    'Nombre de patients': total_in_category,
                    'Pourcentage du total (%)': (total_in_category / total_patients) * 100,
                    'Nombre de réadmissions': readmitted_count,
                    'Taux de réadmission (%)': (readmitted_count / total_in_category) * 100
                }
                diabetes_table.append(row)
    
    diabetes_table_df = pd.DataFrame(diabetes_table)
    diabetes_table_df = diabetes_table_df.sort_values(by='Taux de réadmission (%)', ascending=False)
    
    # Appliquer un style moderne au tableau
    styled_table = diabetes_table_df.style.background_gradient(
        subset=['Taux de réadmission (%)'],
        cmap='Reds'
    ).format({
        'Pourcentage du total (%)': '{:.1f}%',
        'Taux de réadmission (%)': '{:.1f}%'
    })
    
    st.dataframe(styled_table, use_container_width=True)
    
    # Ajouter une boîte d'information
    st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> Les taux de réadmission sont calculés en pourcentage du nombre total de patients dans chaque catégorie.
            Les valeurs plus élevées indiquent un risque de réadmission plus important.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#=============================================================================
# PAGE 5: HISTORIQUE MÉDICAL
#=============================================================================
elif page == "Historique Médical":
    # Style moderne pour la page historique médical
    st.markdown("""
        <style>
        .history-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
            background: linear-gradient(45deg, #1f77b4, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .history-section {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .history-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-box {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        .variable-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="history-header">📋 Analyse de l\'Historique Médical</h1>', unsafe_allow_html=True)

    # Section 1: Vue d'ensemble
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="history-title">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    # Créer 3 colonnes pour les métriques clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_outpatient = df['n_outpatient'].mean()
        st.metric("Consultations Externes Moyennes", f"{avg_outpatient:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_inpatient = df['n_inpatient'].mean()
        st.metric("Hospitalisations Antérieures Moyennes", f"{avg_inpatient:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_emergency = df['n_emergency'].mean()
        st.metric("Visites aux Urgences Moyennes", f"{avg_emergency:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Analyse Détaillée
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="history-title">🔍 Analyse Détaillée</h2>', unsafe_allow_html=True)
    
    # Sélecteur de variable avec style moderne
    st.markdown('<div class="variable-selector">', unsafe_allow_html=True)
    visit_vars = ['n_outpatient', 'n_inpatient', 'n_emergency']
    selected_var = st.selectbox(
        "Sélectionnez une variable d'historique médical à analyser:",
        visit_vars,
        format_func=lambda x: {
            'n_outpatient': 'Consultations Externes',
            'n_inpatient': 'Hospitalisations Antérieures',
            'n_emergency': 'Visites Urgences'
        }[x]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de la variable
        fig = px.histogram(
            df, 
            x=selected_var, 
            nbins=20,
            title=f'Distribution de {selected_var}',
            color_discrete_sequence=['royalblue'],
            opacity=0.7
        )
        fig.update_layout(
            xaxis_title={
                'n_outpatient': 'Nombre de consultations externes',
                'n_inpatient': 'Nombre d\'hospitalisations antérieures',
                'n_emergency': 'Nombre de visites aux urgences'
            }[selected_var],
            yaxis_title="Nombre de patients",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Boxplot par statut de réadmission
        fig = px.box(
            df, 
            x='readmitted', 
            y=selected_var, 
            title=f'{selected_var} par Statut de Réadmission',
            color='readmitted',
            color_discrete_map={'yes': 'firebrick', 'no': 'royalblue'},
            points="all"
        )
        fig.update_layout(
            xaxis_title="Réadmission", 
            yaxis_title={
                'n_outpatient': 'Nombre de consultations externes',
                'n_inpatient': 'Nombre d\'hospitalisations antérieures',
                'n_emergency': 'Nombre de visites aux urgences'
            }[selected_var],
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques descriptives avec style moderne
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="history-title">📊 Statistiques Descriptives</h3>', unsafe_allow_html=True)
    
    stats = df.groupby('readmitted')[selected_var].describe().reset_index()
    stats_style = stats.style.background_gradient(cmap='Blues')
    st.dataframe(stats_style, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Impact des Visites Antérieures
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="history-title">⚠️ Impact des Visites Antérieures</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for i, (col, var) in enumerate(zip([col1, col2, col3], visit_vars)):
        with col:
            # Créer une variable binaire (avec/sans visites)
            has_visits = (df[var] > 0).astype(int)
            visit_status = has_visits.map({0: 'Sans', 1: 'Avec'})
            
            # Créer un DataFrame propre pour le graphique
            visit_data = pd.DataFrame({
                'Statut': visit_status,
                'Readmitted': df['readmitted']
            })
            
            # Calculer manuellement les taux de réadmission
            readmit_rates = []
            for status in ['Sans', 'Avec']:
                subset = visit_data[visit_data['Statut'] == status]
                if len(subset) > 0:
                    rate = (subset['Readmitted'] == 'yes').mean() * 100
                    readmit_rates.append({'Statut': status, 'Taux': rate})
            
            # Créer un DataFrame des taux
            if readmit_rates:
                rates_df = pd.DataFrame(readmit_rates)
                
                var_title = {
                    'n_outpatient': 'Consultations Externes',
                    'n_inpatient': 'Hospitalisations',
                    'n_emergency': 'Visites Urgences'
                }[var]
                
                fig = px.bar(
                    rates_df, 
                    x='Statut', 
                    y='Taux', 
                    title=f'Réadmission: {var_title}',
                    labels={'Statut': var_title, 'Taux': 'Taux (%)'},
                    color='Taux',
                    color_continuous_scale='Reds',
                    text_auto=True
                )
                fig.update_layout(
                    xaxis_title=var_title,
                    yaxis_title="Taux (%)",
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Analyse Multivariée
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="history-title">🔄 Analyse Multivariée</h2>', unsafe_allow_html=True)
    
    # Créer des variables binaires pour chaque type de visite
    df_hist = df.copy()
    for var in visit_vars:
        df_hist[f'has_{var}'] = (df_hist[var] > 0).astype(int)
    
    # Combiner les 3 types de visites
    df_hist['visit_combo'] = (
        df_hist['has_n_outpatient'].astype(str) + '_' + 
        df_hist['has_n_inpatient'].astype(str) + '_' + 
        df_hist['has_n_emergency'].astype(str)
    )
    
    # Créer un mapping plus lisible
    combo_mapping = {
        '0_0_0': 'Aucune visite',
        '1_0_0': 'Seulement consultation',
        '0_1_0': 'Seulement hospitalisation',
        '0_0_1': 'Seulement urgence',
        '1_1_0': 'Consultation + hospitalisation',
        '1_0_1': 'Consultation + urgence',
        '0_1_1': 'Hospitalisation + urgence',
        '1_1_1': 'Tous types de visites'
    }
    
    df_hist['visit_pattern'] = df_hist['visit_combo'].map(combo_mapping)
    
    # Calculer les taux de réadmission pour chaque pattern
    patterns = df_hist['visit_pattern'].unique()
    pattern_data = []
    
    for pattern in patterns:
        subset = df_hist[df_hist['visit_pattern'] == pattern]
        count = len(subset)
        if count > 0:
            rate = (subset['readmitted'] == 'yes').mean() * 100
            pattern_data.append({
                'Pattern': pattern,
                'Taux': rate,
                'Nombre': count
            })
    
    # Créer un DataFrame et trier par taux de réadmission
    if pattern_data:
        pattern_df = pd.DataFrame(pattern_data)
        pattern_df = pattern_df.sort_values(by='Taux', ascending=False)
        
        # Créer le graphique
        fig = px.bar(
            pattern_df, 
            x='Pattern', 
            y='Taux', 
            title='Taux de Réadmission par Combinaison de Visites Antérieures',
            labels={'Pattern': 'Combinaison de visites', 'Taux': 'Taux de réadmission (%)'},
            color='Taux',
            color_continuous_scale='Reds',
            hover_data=['Nombre'],
            text_auto=True
        )
        fig.update_layout(
            xaxis_title="Combinaison de visites",
            yaxis_title="Taux de réadmission (%)",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Ajouter une boîte d'information
        st.markdown("""
            <div class="info-box">
                <strong>Note:</strong> Cette analyse montre l'impact combiné des différents types de visites médicales antérieures sur le taux de réadmission.
                Les combinaisons avec des taux plus élevés indiquent des patterns de visites associés à un risque de réadmission plus important.
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#=============================================================================
# PAGE 6: CORRÉLATIONS
#=============================================================================
elif page == "Corrélations":
    # Style moderne pour la page corrélations
    st.markdown("""
        <style>
        .correlation-header {
            font-size: 2.5em;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1em;
            background: linear-gradient(45deg, #1f77b4, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .correlation-section {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .correlation-title {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-box {
            background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        .variable-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .correlation-strength {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            text-align: center;
            font-weight: bold;
        }
        .strong-correlation {
            background-color: #ffebee;
            color: #c62828;
        }
        .moderate-correlation {
            background-color: #fff3e0;
            color: #ef6c00;
        }
        .weak-correlation {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête principal
    st.markdown('<h1 class="correlation-header">🔗 Analyse des Corrélations</h1>', unsafe_allow_html=True)

    # Section 1: Vue d'ensemble
    st.markdown('<div class="correlation-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="correlation-title">📊 Vue d\'Ensemble</h2>', unsafe_allow_html=True)
    
    # Variables numériques pour la matrice de corrélation
    numeric_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 
                    'n_outpatient', 'n_inpatient', 'n_emergency']
    
    # Créer 3 colonnes pour les métriques clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        max_corr = df[numeric_cols].corr().abs().max().max()
        st.metric("Corrélation Maximale", f"{max_corr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        min_corr = df[numeric_cols].corr().abs().min().min()
        st.metric("Corrélation Minimale", f"{min_corr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_corr = df[numeric_cols].corr().abs().mean().mean()
        st.metric("Corrélation Moyenne", f"{avg_corr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Matrice de Corrélation
    st.markdown('<div class="correlation-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="correlation-title">🔄 Matrice de Corrélation</h2>', unsafe_allow_html=True)
    
    # Matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="auto",
        title='Corrélation entre Variables Numériques'
    )
    fig.update_layout(
        template='plotly_white',
        height=800,
        width=800
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Ajouter une boîte d'information
    st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> La matrice de corrélation montre les relations entre les variables numériques.
            Les valeurs proches de 1 (rouge) indiquent une forte corrélation positive,
            proches de -1 (bleu) une forte corrélation négative,
            et proches de 0 (blanc) une faible corrélation.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Corrélations Significatives
    st.markdown('<div class="correlation-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="correlation-title">⚠️ Corrélations Significatives</h2>', unsafe_allow_html=True)
    
    sig_corrs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.2:
                sig_corrs.append({
                    'Variable 1': numeric_cols[i],
                    'Variable 2': numeric_cols[j],
                    'Corrélation': corr
                })
    
    sig_corrs_df = pd.DataFrame(sig_corrs)
    if not sig_corrs_df.empty:
        sig_corrs_df = sig_corrs_df.sort_values(by='Corrélation', ascending=False)
        
        # Appliquer un style moderne au tableau
        styled_table = sig_corrs_df.style.background_gradient(
            subset=['Corrélation'],
            cmap='RdBu_r',
            vmin=-1,
            vmax=1
        ).format({
            'Corrélation': '{:.3f}'
        })
        
        st.dataframe(styled_table, use_container_width=True)
    else:
        st.markdown("""
            <div class="info-box">
                Aucune corrélation significative (|r| > 0.2) n'a été trouvée entre les variables.
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Analyse Bivariée
    st.markdown('<div class="correlation-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="correlation-title">📈 Analyse Bivariée</h2>', unsafe_allow_html=True)
    
    # Sélecteur de variables avec style moderne
    st.markdown('<div class="variable-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "Variable X",
            numeric_cols,
            format_func=lambda x: {
                'time_in_hospital': 'Durée d\'Hospitalisation',
                'n_lab_procedures': 'Nombre d\'Examens Labo',
                'n_procedures': 'Nombre de Procédures',
                'n_medications': 'Nombre de Médicaments',
                'n_outpatient': 'Consultations Externes',
                'n_inpatient': 'Hospitalisations',
                'n_emergency': 'Visites Urgences'
            }[x]
        )
    
    with col2:
        y_var = st.selectbox(
            "Variable Y",
            numeric_cols,
            index=1,
            format_func=lambda x: {
                'time_in_hospital': 'Durée d\'Hospitalisation',
                'n_lab_procedures': 'Nombre d\'Examens Labo',
                'n_procedures': 'Nombre de Procédures',
                'n_medications': 'Nombre de Médicaments',
                'n_outpatient': 'Consultations Externes',
                'n_inpatient': 'Hospitalisations',
                'n_emergency': 'Visites Urgences'
            }[x]
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scatter plot
    fig = px.scatter(
        df, 
        x=x_var, 
        y=y_var, 
        color='readmitted',
        opacity=0.7,
        color_discrete_map={'yes': 'firebrick', 'no': 'royalblue'},
        title=f'Relation entre {x_var} et {y_var}',
        trendline='ols'
    )
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title=y_var,
        legend_title="Réadmission",
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher la corrélation avec style
    correlation = corr_matrix.loc[x_var, y_var]
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Coefficient de Corrélation", f"{correlation:.3f}")
    
    # Afficher la force de la corrélation avec style
    if abs(correlation) > 0.7:
        strength_class = "strong-correlation"
        strength_text = "Corrélation Forte"
    elif abs(correlation) > 0.4:
        strength_class = "moderate-correlation"
        strength_text = "Corrélation Modérée"
    else:
        strength_class = "weak-correlation"
        strength_text = "Corrélation Faible"
    
    st.markdown(f'<div class="correlation-strength {strength_class}">{strength_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ajouter une boîte d'information
    st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> Le graphique montre la relation entre les deux variables sélectionnées,
            avec une ligne de tendance (OLS) pour visualiser la corrélation.
            Les points sont colorés selon le statut de réadmission pour identifier d'éventuels patterns.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Footer
#st.markdown("---")
#st.markdown("Dashboard Projet Data Analysis, Dataset: hospital_readmissions.csv")
#st.markdown("Analyse des Réadmissions Hospitalières") 