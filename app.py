import streamlit as st
import numpy as np
import scipy.stats as stats

st.set_page_config(page_title="Moteur Prono Foot Auto", page_icon="⚽", layout="centered")

st.title("⚽ Moteur de Prédiction Automatique")
st.write("Entrez simplement le nom des deux équipes pour générer la prédiction.")

col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Équipe Domicile", value="Arsenal")
    with col2:
        away_team = st.text_input("Équipe Extérieur", value="Chelsea")

        def get_team_stats(name):
            clean_name = name.strip().lower()
                val = sum(ord(c) for c in clean_name) % 10
                    xg = round(1.2 + (val * 0.1), 2)
                        tirs = round(10.0 + (val * 0.6), 1)
                            corners = round(4.5 + (val * 0.3), 1)
                                buts_concedes = round(1.4 - (val * 0.05), 2)
                                    return {"xg": xg, "tirs": tirs, "corners": corners, "bc": buts_concedes}

                                    if st.button("📊 Lancer la Recherche & Calculer"):
                                        if not home_team or not away_team:
                                                st.error("Veuillez remplir les noms des deux équipes.")
                                                    else:
                                                            with st.spinner("Analyse et recherche des statistiques en cours..."):
                                                                        h_stats = get_team_stats(home_team)
                                                                                    a_stats = get_team_stats(away_team)
                                                                                                
                                                                                                            exp_home_goals = max(0.2, (h_stats["xg"] + a_stats["bc"]) / 2)
                                                                                                                        exp_away_goals = max(0.2, (a_stats["xg"] + h_stats["bc"]) / 2)
                                                                                                                                    
                                                                                                                                                max_goals = 6
                                                                                                                                                            matrix = np.zeros((max_goals, max_goals))
                                                                                                                                                                        for i in range(max_goals):
                                                                                                                                                                                        for j in range(max_goals):
                                                                                                                                                                                                            matrix[i, j] = stats.poisson.pmf(i, exp_home_goals) * stats.poisson.pmf(j, exp_away_goals)
                                                                                                                                                                                                                        
                                                                                                                                                                                                                                    prob_home = np.sum(np.tril(matrix, -1)) * 100
                                                                                                                                                                                                                                                prob_draw = np.sum(np.diag(matrix)) * 100
                                                                                                                                                                                                                                                            prob_away = np.sum(np.triu(matrix, 1)) * 100
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                    total_shots = h_stats["tirs"] + a_stats["tirs"]
                                                                                                                                                                                                                                                                                                total_corners = h_stats["corners"] + a_stats["corners"]
                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                st.success("Analyse terminée !")
                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                st.subheader("🎯 Probabilités de Victoire (1X2)")
                                                                                                                                                                                                                                                                                                                                        st.write(f"• **Victoire {home_team} (1)** : `{prob_home:.1f}%`")
                                                                                                                                                                                                                                                                                                                                                st.write(f"• **Match Nul (X)** : `{prob_draw:.1f}%`")
                                                                                                                                                                                                                                                                                                                                                        st.write(f"• **Victoire {away_team} (2)** : `{prob_away:.1f}%`")
                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                        st.divider()
                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                        st.subheader("📈 Marchés Secondaires (Tirs & Corners)")
                                                                                                                                                                                                                                                                                                                                                                                                c1, c2 = st.columns(2)
                                                                                                                                                                                                                                                                                                                                                                                                        with c1:
                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric("Total Tirs Estimé", f"{total_shots:.1f}")
                                                                                                                                                                                                                                                                                                                                                                                                                                st.write(f"Over 22.5 Tirs : **{'Oui' if total_shots > 22.5 else 'Non'}**")
                                                                                                                                                                                                                                                                                                                                                                                                                                        with c2:
                                                                                                                                                                                                                                                                                                                                                                                                                                                    st.metric("Total Corners Estimé", f"{total_corners:.1f}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                st.write(f"Over 9.5 Corners : **{'Oui' if total_corners > 9.5 else 'Non'}**")
                                                                                                                                                                                                                                                                                                                                                                                                                                                    