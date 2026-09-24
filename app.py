import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="PronoEngine Mobile", layout="centered")

st.title("⚽ Moteur de Prédiction Football")

# Entrées utilisateur
home_team = st.text_input("Équipe Domicile", "Arsenal")
away_team = st.text_input("Équipe Extérieur", "Chelsea")

st.subheader("Moyennes des 5 derniers matchs")
col1, col2 = st.columns(2)

with col1:
    h_xg = st.number_input("xG Domicile", value=1.85, step=0.1)
    h_shots = st.number_input("Tirs Domicile", value=14.2, step=0.5)
    h_corners = st.number_input("Corners Domicile", value=6.5, step=0.5)

with col2:
    a_xg = st.number_input("xG Extérieur", value=1.10, step=0.1)
    a_shots = st.number_input("Tirs Extérieur", value=9.5, step=0.5)
    a_corners = st.number_input("Corners Extérieur", value=4.1, step=0.5)

if st.button("📊 Calculer la Prédiction", use_container_width=True):
    # Calculations
    win_home = 0
    draw = 0
    win_away = 0
    for h in range(6):
        for a in range(6):
            prob = poisson.pmf(h, h_xg) * poisson.pmf(a, a_xg)
            if h > a: win_home += prob
            elif h == a: draw += prob
            else: win_away += prob

    h_sot = round(h_shots * 0.35, 1)
    a_sot = round(a_shots * 0.32, 1)
    total_corners = h_corners + a_corners
    prob_under10 = sum([poisson.pmf(k, total_corners) for k in range(10)])
    prob_over95_corners = round((1 - prob_under10) * 100, 1)

    # Affichage
    st.markdown("---")
    st.success("### Résultats")
    
    st.write(f"**Score xG prévu :** {h_xg} - {a_xg}")
    st.write(f"**Probabilités :** {home_team} ({win_home*100:.0f}%) | Nul ({draw*100:.0f}%) | {away_team} ({win_away*100:.0f}%)")
    
    st.write(f"**Tirs totaux estimés :** {h_shots + a_shots:.1f} ({home_team}: {h_shots} | {away_team}: {a_shots})")
    st.write(f"**Tirs cadrés estimés :** {h_sot + a_sot:.1f} ({home_team}: {h_sot} | {away_team}: {a_sot})")
    st.write(f"**Corners totaux :** {total_corners:.1f} (Over 9.5 Corners : **{prob_over95_corners}%**)")
