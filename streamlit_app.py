import numpy as np
from scipy.stats import poisson

def parse_matches(match_string):
    """
    Parse une chaîne de caractères au format "A-B, C-D, ..." 
    où chaque élément représente (Buts_Marqués - Buts_Encaissés).
    Les matchs doivent être ordonnés du plus récent au plus ancien.
    """
    matches = match_string.split(',')
    scored = []
    conceded = []
    for m in matches:
        m = m.strip()
        try:
            parts = m.split('-')
            if len(parts) == 2:
                scored.append(int(parts[0].strip()))
                conceded.append(int(parts[1].strip()))
        except ValueError:
            continue
    return scored, conceded

def calculate_exponential_weights(n, decay_factor=0.85):
    """
    Génère des poids décroissants exponentiellement pour les n derniers matchs.
    L'index 0 (match le plus récent) reçoit le poids le plus élevé.
    """
    weights = [decay_factor ** i for i in range(n)]
    return np.array(weights) / sum(weights)

def predict_fifa_match(team_a_name, team_a_str, team_b_name, team_b_str, max_goals=5):
    """
    Calcule de manière déterministe les 3 scores exacts les plus probables 
    pour un match entre l'équipe A et l'équipe B.
    """
    # 1. Extraction des données
    scored_a, conceded_a = parse_matches(team_a_str)
    scored_b, conceded_b = parse_matches(team_b_str)
    
    if not scored_a or not scored_b:
        raise ValueError("Format des données d'entrée invalide ou vide.")

    # 2. Application de la pondération temporelle
    weights_a = calculate_exponential_weights(len(scored_a))
    weights_b = calculate_exponential_weights(len(scored_b))

    # Moyennes pondérées pour l'attaque et la défense de chaque équipe
    att_a = np.sum(np.array(scored_a) * weights_a)
    def_a = np.sum(np.array(conceded_a) * weights_a)
    
    att_b = np.sum(np.array(scored_b) * weights_b)
    def_b = np.sum(np.array(conceded_b) * weights_b)

    # 3. Estimation des taux d'intensité (Expected Goals / xG)
    # λ_A (buts attendus pour A) = Force offensive de A croisée avec la faiblesse défensive de B
    lambda_a = (att_a + def_b) / 2.0
    lambda_b = (att_b + def_a) / 2.0

    # 4. Génération de la matrice de probabilités de Poisson croisées
    # On calcule P(X = i) pour l'équipe A et P(Y = j) pour l'équipe B
    grid_size = max_goals + 1
    matrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            prob_a = poisson.pmf(i, lambda_a)
            prob_b = poisson.pmf(j, lambda_b)
            matrix[i, j] = prob_a * prob_b

    # Normalisation de la matrice sur la grille considérée pour sommer à 100%
    matrix /= np.sum(matrix)

    # 5. Extraction et tri des scores
    scores_probabilities = []
    for i in range(grid_size):
        for j in range(grid_size):
            score_label = f"{team_a_name} {i} - {j} {team_b_name}"
            prob_percent = matrix[i, j] * 100.0
            scores_probabilities.append((score_label, prob_percent))

    # Tri décroissant selon la probabilité
    scores_probabilities.sort(key=lambda x: x[1], reverse=True)

    # Retourne les prédictions et les xG estimés
    return lambda_a, lambda_b, scores_probabilities[:3]

# ==========================================
# EXEMPLE CONCRET D'UTILISATION
# ==========================================
if __name__ == "__main__":
    # Formats : "ButsMarqués-ButsEncaissés" (du plus récent au plus ancien)
    # Équipe A (ex: Real Madrid - Virtuel)
    form_equipe_a = "3-1, 2-2, 4-0, 1-1, 2-0"
    
    # Équipe B (ex: Manchester City - Virtuel)
    form_equipe_b = "2-2, 1-0, 3-2, 0-1, 2-1"

    nom_a = "Real Madrid"
    nom_b = "Man City"

    xG_a, xG_b, top_3_scores = predict_fifa_match(nom_a, form_equipe_a, nom_b, form_equipe_b)

    print(f"--- ANALYSE STATISTIQUE DU MATCH : {nom_a} vs {nom_b} ---")
    print(f"Buts attendus (xG) -> {nom_a}: {xG_a:.2f} | {nom_b}: {xG_b:.2f}\n")
    print("Top 3 des scores exacts les plus probables :")
    print("-" * 45)
    
    for rank, (score, prob) in enumerate(top_3_scores, 1):
        print(f"{rank}. {score}  --->  {prob:.2f}%")
    
