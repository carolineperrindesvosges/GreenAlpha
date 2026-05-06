# GreenAlpha

## Présentation
**GreenAlpha Challenge** est un serious game pédagogique en finance durable conçu pour des étudiants de **M1 Finance**.  
L’application a pour objectif de placer les étudiants en situation de décision dans un contexte de gestion de portefeuille, en articulant les notions de **rendement**, **risque** et **contrainte ESG**.

Le dispositif repose sur une logique de **simulation** : les étudiants construisent d’abord un portefeuille initial, puis doivent réagir à un choc de marché, avant de proposer une nouvelle allocation compatible avec des exigences de soutenabilité.  
L’ensemble vise à renforcer à la fois la compréhension des concepts financiers fondamentaux et la capacité à justifier des arbitrages d’investissement dans un univers incertain.

---

## Objectifs pédagogiques
Cette application permet notamment de travailler les compétences suivantes :

- comprendre les principes de base de l’allocation de portefeuille ;
- raisonner en termes de compromis entre rentabilité et risque ;
- intégrer une contrainte ESG dans une décision d’investissement ;
- interpréter les effets d’un choc de marché sur une allocation ;
- développer une capacité d’argumentation et de justification des choix financiers.

---

## Public visé
Le serious game est destiné en priorité à des étudiants de :

- **M1 Finance**
- formation initiale ou alternance
- avec des prérequis de base en finance et en statistiques

---

## Déroulement du serious game
L’application est structurée en plusieurs temps :

1. **Construction d’un portefeuille initial**  
   Les étudiants choisissent une allocation entre plusieurs actifs.

2. **Observation d’un choc de marché**  
   Un événement exogène modifie les conditions de marché et oblige à reconsidérer les choix initiaux.

3. **Réallocation sous contrainte ESG**  
   Les étudiants doivent adapter leur portefeuille en intégrant une exigence de score ESG minimal.

4. **Évaluation finale**  
   L’application permet de visualiser les résultats obtenus et d’alimenter une discussion ou une restitution orale.

---

## Intérêt pédagogique
Ce projet s’inscrit dans une démarche de **pédagogie active** et de **mise en situation professionnelle**.  
Il cherche à dépasser le simple exercice technique en proposant une expérience d’apprentissage plus interactive, dans laquelle les étudiants sont amenés à :

- prendre des décisions ;
- observer les conséquences de leurs choix ;
- comparer différentes stratégies ;
- défendre leur allocation comme dans un cadre professionnel.

L’outil peut être utilisé comme support de séance, activité d’application, ou base pour une restitution collective.

---

## Technologies utilisées
L’application est développée en **Python** avec **Streamlit**.

Principales bibliothèques mobilisées :
- streamlit
- pandas
- numpy
- matplotlib

---

## Lancement de l’application en local
Pour exécuter l’application localement :

```bash
pip install -r requirements.txt
streamlit run app.py
