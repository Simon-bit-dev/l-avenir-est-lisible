
import itertools
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Arbre existentiel probabiliste — V2", layout="wide")

LIKERT = {
    "1 — Pas du tout d’accord": 1,
    "2 — Plutôt pas d’accord": 2,
    "3 — Un peu en désaccord": 3,
    "4 — Ni d’accord ni en désaccord": 4,
    "5 — Un peu d’accord": 5,
    "6 — Plutôt d’accord": 6,
    "7 — Tout à fait d’accord": 7,
}

DIMENSIONS = [
    {"key":"valence","label":"Valence affective","positive":"positive","negative":"négative","mixed":"mixte","items":[
        ("Dans l’ensemble, je retire actuellement davantage de satisfaction que d’insatisfaction de ma vie.",False),
        ("Je ressens régulièrement des émotions positives au cours d’une semaine ordinaire.",False),
        ("Ma situation actuelle comporte davantage d’éléments que je souhaite conserver que d’éléments que je voudrais voir disparaître.",False),
        ("Je me sens globalement bien dans la période que je traverse.",False),
        ("Si les prochains mois ressemblaient aux dernières semaines, cela me conviendrait plutôt.",False)]},
    {"key":"stress","label":"Activation / stress","positive":"apaisée","negative":"sous tension","mixed":"mixte","items":[
        ("J’ai fréquemment le sentiment d’avoir trop de choses à gérer en même temps.",True),
        ("Mon esprit reste souvent mobilisé par des préoccupations, même lorsque je pourrais me détendre.",True),
        ("Je me sens généralement calme face aux obligations ordinaires de ma vie.",False),
        ("J’éprouve régulièrement une tension liée à ce qui pourrait mal se passer.",True),
        ("J’ai suffisamment de moments où je me sens véritablement détendu.",False)]},
    {"key":"agency","label":"Agentivité","positive":"active","negative":"passive","mixed":"mixte","items":[
        ("Lorsque quelque chose dans ma vie ne me convient pas, j’ai tendance à agir pour le modifier.",False),
        ("J’ai le sentiment d’exercer une influence réelle sur la direction que prend ma vie.",False),
        ("Je reporte souvent des décisions importantes alors même que je pourrais les prendre.",True),
        ("Face à une difficulté, je cherche généralement une manière concrète d’avancer.",False),
        ("J’ai parfois l’impression que ma situation dépend principalement de circonstances sur lesquelles je n’ai aucune prise.",True)]},
    {"key":"social","label":"Appartenance sociale","positive":"intégrée","negative":"isolée","mixed":"mixte","items":[
        ("Je dispose de plusieurs personnes vers lesquelles je peux réellement me tourner en cas de besoin.",False),
        ("Je me sens intégré à un ou plusieurs groupes auxquels j’accorde de l’importance.",False),
        ("Il m’arrive souvent de me sentir seul, même lorsque d’autres personnes sont présentes autour de moi.",True),
        ("Mes relations actuelles me donnent le sentiment de compter pour certaines personnes.",False),
        ("J’ai parfois le sentiment d’être en marge de la vie sociale qui m’entoure.",True)]},
    {"key":"material","label":"Stabilité matérielle","positive":"stable","negative":"instable","mixed":"mixte","items":[
        ("Ma situation financière actuelle me permet d’envisager les prochains mois sans inquiétude majeure.",False),
        ("Mon logement ou ma situation résidentielle me paraît suffisamment stable.",False),
        ("Ma situation professionnelle ou académique actuelle me paraît relativement sécurisée.",False),
        ("Un événement imprévu important pourrait rapidement fragiliser ma situation matérielle.",True),
        ("J’ai une visibilité raisonnable sur mes conditions de vie dans les douze prochains mois.",False)]},
    {"key":"progress","label":"Progression perçue","positive":"en progression","negative":"en stagnation / régression","mixed":"mixte","items":[
        ("J’ai le sentiment d’avancer vers des objectifs qui comptent réellement pour moi.",False),
        ("Ma situation actuelle me semble meilleure qu’il y a un an sur plusieurs dimensions importantes.",False),
        ("J’ai parfois l’impression de faire beaucoup d’efforts sans réellement progresser.",True),
        ("Je développe actuellement des compétences, des relations ou des ressources susceptibles d’améliorer ma situation future.",False),
        ("J’ai le sentiment que ma trajectoire générale va dans une direction qui me convient.",False)]},
    {"key":"coherence","label":"Cohérence existentielle","positive":"cohérente","negative":"conflictuelle","mixed":"mixte","items":[
        ("Mes choix actuels correspondent assez bien à ce que je considère important dans la vie.",False),
        ("Il existe actuellement un écart important entre la vie que je mène et celle que je voudrais mener.",True),
        ("Les différentes parties de ma vie — travail, relations, projets, valeurs — s’accordent plutôt bien entre elles.",False),
        ("Je poursuis parfois des objectifs dont je ne suis plus certain qu’ils soient réellement les miens.",True),
        ("Je comprends assez clairement pourquoi je fais ce que je fais actuellement.",False)]},
    {"key":"future","label":"Projection temporelle","positive":"confiante","negative":"inquiète / pessimiste","mixed":"mixte","items":[
        ("Lorsque je pense à mon avenir, j’envisage davantage de possibilités désirables que de menaces.",False),
        ("Je pense que ma situation a de bonnes chances de s’améliorer au cours des prochaines années.",False),
        ("J’ai du mal à imaginer une version future de ma vie qui me satisfasse réellement.",True),
        ("J’ai plusieurs projets ou possibilités futures qui m’enthousiasment.",False),
        ("L’avenir me paraît davantage incertain de manière inquiétante qu’ouvert de manière stimulante.",True)]},
]

CENTERS = {"-":-0.65,"±":0.0,"+":0.65}
SIGMA = 0.35

def item_score(answer, reverse):
    z=(answer-4)/3
    return -z if reverse else z

def membership(score):
    w={k:math.exp(-((score-mu)**2)/(2*SIGMA**2)) for k,mu in CENTERS.items()}
    s=sum(w.values())
    return {k:v/s for k,v in w.items()}

def state_label(dim,state):
    return {"+":dim["positive"],"-":dim["negative"],"±":dim["mixed"]}[state]

def expected_latent(probs):
    return probs["+"] - probs["-"]

def entropy_3(probs):
    vals=[p for p in probs.values() if p>0]
    return -sum(p*math.log(p,3) for p in vals)

# Context factors
CONTEXT_OPTIONS = {
    "employment": ("Situation principale", {
        "Études":0.05, "Emploi salarié stable":0.55, "Emploi salarié précaire ou temporaire":-0.35,
        "Activité indépendante":0.05, "Création d’entreprise":-0.10, "Recherche d’emploi":-0.60,
        "Sans activité professionnelle ou académique":-0.55, "Retraite":0.20, "Autre":0.0}),
    "finances": ("Situation financière", {
        "Très confortable":1.0, "Confortable":0.5, "Correcte mais contrainte":0.0, "Fragile":-0.5, "Très fragile":-1.0}),
    "housing": ("Environnement de vie", {
        "Me convient beaucoup":1.0, "Me convient plutôt":0.5, "Moyennement":0.0, "Me convient peu":-0.5, "Ne me convient pas du tout":-1.0}),
    "relationship": ("Situation sentimentale", {
        "Célibataire":0.0, "Relation non cohabitante":0.25, "Relation cohabitante":0.45,
        "Marié ou pacsé":0.50, "Relation en cours de rupture ou très instable":-0.75, "Autre":0.0}),
    "health": ("État de santé général", {
        "Très bon":1.0, "Bon":0.5, "Moyen":0.0, "Mauvais":-0.5, "Très mauvais":-1.0}),
    "change": ("Changement majeur envisagé dans les 12 mois", {"Oui":0.20, "Peut-être":0.10, "Non":0.0}),
}

EVENT_EFFECT = {
    "Aucun événement majeur":0.0,
    "Promotion / réussite importante":0.65,
    "Début d’une relation importante":0.45,
    "Déménagement positif":0.30,
    "Rupture":-0.55,
    "Perte d’emploi":-0.70,
    "Problème de santé important":-0.70,
    "Difficulté financière majeure":-0.65,
    "Échec personnel important":-0.45,
    "Autre / ambivalent":0.0,
}

# Per-dimension sensitivity matrix: context does not act uniformly.
SENS = {
    "valence":      {"employment":0.10,"finances":0.10,"housing":0.08,"relationship":0.16,"health":0.18,"change":0.05,"event":0.16},
    "stress":       {"employment":0.12,"finances":0.15,"housing":0.08,"relationship":0.10,"health":0.12,"change":-0.08,"event":0.16},
    "agency":       {"employment":0.05,"finances":0.04,"housing":0.02,"relationship":0.02,"health":0.08,"change":0.10,"event":0.05},
    "social":       {"employment":0.03,"finances":0.02,"housing":0.05,"relationship":0.30,"health":0.05,"change":0.02,"event":0.14},
    "material":     {"employment":0.28,"finances":0.34,"housing":0.18,"relationship":0.02,"health":0.04,"change":-0.05,"event":0.18},
    "progress":     {"employment":0.16,"finances":0.08,"housing":0.02,"relationship":0.02,"health":0.08,"change":0.16,"event":0.12},
    "coherence":    {"employment":0.06,"finances":0.02,"housing":0.03,"relationship":0.08,"health":0.05,"change":0.10,"event":0.08},
    "future":       {"employment":0.14,"finances":0.12,"housing":0.06,"relationship":0.08,"health":0.12,"change":0.12,"event":0.16},
}

def future_probs(current_probs, context, dim_key):
    cur=expected_latent(current_probs)
    shift=0.0
    for k,w in SENS[dim_key].items():
        shift += w*context.get(k,0.0)
    # persistence remains dominant
    projected=max(-1,min(1,0.74*cur+shift))
    return membership(projected), projected

def detect_tensions(scores):
    t=[]
    def add(title, desc, strength):
        if strength>0.12:
            t.append((strength,title,desc))
    add("Progression sous tension",
        "La trajectoire paraît ascendante, mais son coût subjectif est élevé.",
        max(0,scores["progress"])*max(0,-scores["stress"]))
    add("Stabilité sans cohérence",
        "La situation fonctionne objectivement mieux qu’elle ne semble correspondre aux valeurs ou au sens recherché.",
        max(0,scores["material"])*max(0,-scores["coherence"]))
    add("Agentivité sans apaisement",
        "La capacité d’action est forte, mais elle ne s’accompagne pas d’un sentiment de calme.",
        max(0,scores["agency"])*max(0,-scores["stress"]))
    add("Confiance malgré une valence faible",
        "L’avenir est perçu plus positivement que l’expérience présente.",
        max(0,scores["future"])*max(0,-scores["valence"]))
    add("Satisfaction malgré faible progression",
        "Le présent est relativement positif alors que la trajectoire est perçue comme stagnante.",
        max(0,scores["valence"])*max(0,-scores["progress"]))
    add("Intégration sociale sans cohérence",
        "Le soutien relationnel est présent, mais il ne résout pas les conflits de direction existentielle.",
        max(0,scores["social"])*max(0,-scores["coherence"]))
    return sorted(t, reverse=True)[:3]

def scenario_distribution(current, future):
    # Aggregate eight expected latents into three narrative scenario scores.
    now = {k:expected_latent(v["probs"]) for k,v in current.items()}
    fut = {k:expected_latent(v) for k,v in future.items()}

    avg_now=sum(now.values())/8
    avg_fut=sum(fut.values())/8
    stress_fut=fut["stress"]
    coherence_fut=fut["coherence"]
    progress_fut=fut["progress"]

    favorable = 1.2*avg_fut + 0.35*coherence_fut + 0.25*progress_fut + 0.20*stress_fut
    central = 0.6 - abs(avg_fut-avg_now) + 0.15*progress_fut - 0.10*abs(stress_fut)
    adverse = -1.0*avg_fut - 0.25*coherence_fut - 0.20*progress_fut - 0.20*stress_fut

    raw={"Favorable":math.exp(favorable),"Central":math.exp(central),"Défavorable":math.exp(adverse)}
    total=sum(raw.values())
    return {k:v/total for k,v in raw.items()}

def scenario_text(current_scores, future_scores):
    stress=future_scores["stress"]
    prog=future_scores["progress"]
    coh=future_scores["coherence"]
    val=future_scores["valence"]
    if prog>0.25 and stress< -0.15:
        central="poursuite de la progression, mais sous tension persistante"
    elif prog>0.25 and stress>=-0.15:
        central="progression maintenue avec un meilleur équilibre subjectif"
    elif val>0.2:
        central="stabilité globalement favorable sans transformation majeure"
    else:
        central="stabilité fragile, avec peu d’amélioration structurelle"

    favorable="amélioration de l’équilibre général, notamment par hausse de la cohérence et baisse des facteurs de tension"
    adverse="érosion progressive de la satisfaction sous l’effet des fragilités les plus présentes"
    return central,favorable,adverse


def triadic_path_figure(current):
    """
    Visualise la position dominante comme un chemin dans une structure ternaire.
    À chaque profondeur, le parent dominant engendre ses trois possibilités.
    Les alternatives restent visibles, tandis que le chemin choisi est accentué.
    """
    state_order = ["+", "±", "-"]
    branch_code = {"+": "1", "±": "2", "-": "3"}

    edge_x, edge_y = [], []
    alt_x, alt_y, alt_text, alt_hover = [], [], [], []
    dom_x, dom_y, dom_text, dom_hover = [0.0], [8.8], ["R"], ["Racine — état présent"]
    path_x, path_y = [0.0], [8.8]

    parent_x, parent_y = 0.0, 8.8
    path_codes = []
    selected_nodes = []

    for depth, dim in enumerate(DIMENSIONS, start=1):
        probs = current[dim["key"]]["probs"]
        dominant = max(probs, key=probs.get)
        path_codes.append(branch_code[dominant])

        # Decreasing horizontal spread keeps the recursive geometry readable.
        spread = max(0.32, 1.35 * (0.78 ** (depth - 1)))
        y = 8.8 - depth

        child_positions = {
            "+": parent_x - spread,
            "±": parent_x,
            "-": parent_x + spread,
        }

        for state in state_order:
            x = child_positions[state]

            edge_x += [parent_x, x, None]
            edge_y += [parent_y, y, None]

            short = f"{branch_code[state]} · {probs[state]:.0%}"
            hover = (
                f"<b>{dim['label']}</b><br>"
                f"{state_label(dim, state)}<br>"
                f"Probabilité d’appartenance : {probs[state]:.1%}"
            )

            if state == dominant:
                dom_x.append(x)
                dom_y.append(y)
                dom_text.append(short)
                dom_hover.append(hover)
                selected_nodes.append((x, y))
            else:
                alt_x.append(x)
                alt_y.append(y)
                alt_text.append(short)
                alt_hover.append(hover)

        selected_x = child_positions[dominant]
        path_x += [selected_x]
        path_y += [y]
        parent_x, parent_y = selected_x, y

    fig = go.Figure()

    # All locally visible ternary branches.
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="skip",
        showlegend=False,
        line=dict(width=1),
    ))

    # Non-selected sibling branches.
    fig.add_trace(go.Scatter(
        x=alt_x, y=alt_y,
        mode="markers+text",
        text=alt_text,
        textposition="middle right",
        hovertext=alt_hover,
        hoverinfo="text",
        name="Branches alternatives",
        marker=dict(size=9, symbol="circle"),
    ))

    # Dominant path line.
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode="lines",
        hoverinfo="skip",
        name="Chemin dominant",
        line=dict(width=5),
    ))

    # Dominant nodes.
    fig.add_trace(go.Scatter(
        x=dom_x, y=dom_y,
        mode="markers+text",
        text=dom_text,
        textposition="middle left",
        hovertext=dom_hover,
        hoverinfo="text",
        name="Position dominante",
        marker=dict(size=15, symbol="diamond"),
    ))

    # Dimension labels on the left margin.
    label_x = min(alt_x + dom_x) - 0.9
    for depth, dim in enumerate(DIMENSIONS, start=1):
        fig.add_annotation(
            x=label_x,
            y=8.8-depth,
            text=f"{depth}. {dim['label']}",
            showarrow=False,
            xanchor="right",
            font=dict(size=11),
        )

    fig.update_layout(
        title="Position dans l’arbre ternaire — chemin dominant et alternatives locales",
        height=820,
        margin=dict(l=190, r=80, t=70, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0.2, 9.25]),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )

    return fig, ".".join(path_codes)


st.title("Arbre existentiel probabiliste — V3")
st.caption("Le moteur conserve les 6 561 feuilles et situe désormais visuellement le répondant dans la structure ternaire.")

with st.expander("Méthode"):
    st.markdown("""
    - 40 items → 8 scores latents sur `[-1,+1]`
    - chaque dimension → trois probabilités `− / ± / +`
    - 6 561 feuilles calculées en arrière-plan
    - variables factuelles → effets spécifiques selon la dimension
    - sortie → position visuelle dans l’arbre, profil, tensions, scénarios, bifurcations, leviers

    **La projection reste heuristique et non calibrée empiriquement.**
    """)

st.header("1. Questionnaire")

responses={}
q=1
for dim in DIMENSIONS:
    st.subheader(dim["label"])
    for idx,(question,rev) in enumerate(dim["items"]):
        choice=st.radio(f"{q}. {question}", list(LIKERT.keys()), index=3, key=f"{dim['key']}_{idx}")
        responses[(dim["key"],idx)]=LIKERT[choice]
        q+=1

st.header("2. Variables factuelles")
context={}
age=st.number_input("Âge",16,100,30)
for key,(label,options) in CONTEXT_OPTIONS.items():
    choice=st.selectbox(label,list(options.keys()),key=f"ctx_{key}")
    context[key]=options[choice]

event=st.selectbox("Événement principal des 12 derniers mois",list(EVENT_EFFECT.keys()))
context["event"]=EVENT_EFFECT[event]

subjective_improvement=st.slider("Probabilité subjective d’amélioration à 12 mois (%)",0,100,50)
subjective_degradation=st.slider("Probabilité subjective de dégradation à 12 mois (%)",0,100,25)

if st.button("Générer l’analyse", type="primary"):
    current={}
    scores={}
    for dim in DIMENSIONS:
        vals=[item_score(responses[(dim["key"],i)],rev) for i,(_,rev) in enumerate(dim["items"])]
        score=sum(vals)/len(vals)
        probs=membership(score)
        current[dim["key"]]={"score":score,"probs":probs,"dim":dim}
        scores[dim["key"]]=score

    # All 6561 leaves remain in the engine.
    leaves=[]
    for combo in itertools.product(["+","±","-"], repeat=8):
        p=1.0
        for state,dim in zip(combo,DIMENSIONS):
            p*=current[dim["key"]]["probs"][state]
        leaves.append((combo,p))
    leaves.sort(key=lambda x:x[1],reverse=True)

    future={}
    future_scores={}
    for dim in DIMENSIONS:
        fp,sc=future_probs(current[dim["key"]]["probs"],context,dim["key"])
        future[dim["key"]]=fp
        future_scores[dim["key"]]=sc

    st.header("3. Lecture synthétique")

    dom=[]
    for dim in DIMENSIONS:
        state=max(current[dim["key"]]["probs"], key=current[dim["key"]]["probs"].get)
        dom.append(state_label(dim,state))

    avg=sum(scores.values())/8
    if avg>0.30:
        general="configuration globalement favorable"
    elif avg<-0.30:
        general="configuration globalement défavorable"
    else:
        general="configuration globalement mixte"

    strongest_pos=sorted(scores.items(), key=lambda x:x[1], reverse=True)[:3]
    weakest=sorted(scores.items(), key=lambda x:x[1])[:2]
    labelmap={d["key"]:d["label"].lower() for d in DIMENSIONS}

    st.markdown(f"### {general.capitalize()}")
    st.write(
        "Le profil dominant combine "
        + ", ".join(labelmap[k] for k,_ in strongest_pos)
        + ". Les principales zones de fragilité concernent "
        + " et ".join(labelmap[k] for k,_ in weakest)
        + "."
    )
    st.markdown("**Chemin dominant :** " + " → ".join(dom))

    st.subheader("Carte de position dans la structure triadique")
    fig, triadic_code = triadic_path_figure(current)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Coordonnée ontologique dominante :** `{triadic_code}`  "
        "— à chaque niveau : `1 = première branche`, `2 = branche mixte`, `3 = branche opposée`."
    )
    st.caption(
        "La figure montre la région locale pertinente de l’arbre : à chaque niveau, "
        "les trois possibilités sont visibles, mais seule la branche dominante est poursuivie "
        "jusqu’au niveau suivant. L’arbre complet de profondeur 8 contient 9 841 nœuds."
    )

    with st.expander("Voir le détail des 8 dimensions"):
        rows=[]
        for dim in DIMENSIONS:
            p=current[dim["key"]]["probs"]
            rows.append({
                "Dimension":dim["label"],
                "Score":current[dim["key"]]["score"],
                "État dominant":state_label(dim,max(p,key=p.get)),
                "P(+)":p["+"],"P(±)":p["±"],"P(-)":p["-"],
                "Incertitude":entropy_3(p)
            })
        df=pd.DataFrame(rows)
        st.dataframe(df.style.format({"Score":"{:.2f}","P(+)":"{:.1%}","P(±)":"{:.1%}","P(-)":"{:.1%}","Incertitude":"{:.2f}"}),use_container_width=True)
        st.caption(f"Feuille dominante : {leaves[0][1]:.2%} — masse des 10 premières : {sum(p for _,p in leaves[:10]):.2%}")

    st.header("4. Tensions internes")
    tensions=detect_tensions(scores)
    if tensions:
        for _,title,desc in tensions:
            st.markdown(f"**{title}**")
            st.write(desc)
    else:
        st.write("Aucune tension structurelle forte n’est détectée dans ce profil.")

    st.header("5. Hypothèse à 12 mois")
    scen=scenario_distribution(current,future)
    central_text,fav_text,adv_text=scenario_text(scores,future_scores)

    cols=st.columns(3)
    cols[0].metric("Scénario central",f"{scen['Central']:.0%}")
    cols[1].metric("Scénario favorable",f"{scen['Favorable']:.0%}")
    cols[2].metric("Scénario défavorable",f"{scen['Défavorable']:.0%}")

    st.markdown(f"**Central — {scen['Central']:.0%}** : {central_text}.")
    st.markdown(f"**Favorable — {scen['Favorable']:.0%}** : {fav_text}.")
    st.markdown(f"**Défavorable — {scen['Défavorable']:.0%}** : {adv_text}.")

    st.header("6. Bifurcations probables")
    bif=[]
    for dim in DIMENSIONS:
        cur=current[dim["key"]]["probs"]
        fut=future[dim["key"]]
        dominant=max(cur,key=cur.get)
        stay=fut[dominant]
        change=1-stay
        bif.append((change,dim["label"],dominant,fut))
    bif.sort(reverse=True)
    for change,label,dom_state,futp in bif[:3]:
        st.markdown(f"**{label} — {change:.0%} de probabilité de quitter l’état dominant actuel**")
        st.write(f"Distribution projetée : + {futp['+']:.0%} · ± {futp['±']:.0%} · − {futp['-']:.0%}")

    st.header("7. Leviers")
    leverage=[]
    for ctx_key in ["employment","finances","housing","relationship","health","change","event"]:
        impact=sum(abs(SENS[d["key"]].get(ctx_key,0)) for d in DIMENSIONS)
        current_val=context.get(ctx_key,0)
        # leverage = structural impact * distance from best plausible state
        room=1-current_val if current_val>=0 else 1+abs(current_val)
        leverage.append((impact*room,ctx_key,impact,current_val))
    leverage.sort(reverse=True)

    ctx_labels={
        "employment":"situation professionnelle",
        "finances":"situation financière",
        "housing":"environnement de vie",
        "relationship":"situation relationnelle",
        "health":"santé",
        "change":"capacité / probabilité de changement",
        "event":"effet des événements récents"
    }
    for score,key,impact,val in leverage[:3]:
        st.markdown(f"**{ctx_labels[key].capitalize()}**")
        st.write("Fort potentiel de déplacement de la trajectoire dans le modèle actuel.")

    st.header("8. Écart entre modèle et intuition")
    model_improvement=scen["Favorable"]
    model_degradation=scen["Défavorable"]
    st.write(
        f"Votre estimation subjective : amélioration {subjective_improvement}% / dégradation {subjective_degradation}%. "
        f"Le modèle heuristique estime : favorable {model_improvement:.0%} / défavorable {model_degradation:.0%}."
    )
    gap=(subjective_improvement/100)-model_improvement
    if gap>0.15:
        st.info("Votre anticipation est nettement plus optimiste que l’hypothèse produite par le modèle.")
    elif gap<-0.15:
        st.info("Votre anticipation est nettement plus prudente que l’hypothèse produite par le modèle.")
    else:
        st.info("Votre anticipation subjective est relativement proche de l’hypothèse produite par le modèle.")

    st.warning("Projection expérimentale : les probabilités à 12 mois ne sont pas calibrées sur des données longitudinales.")
