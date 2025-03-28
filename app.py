
import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc 
import plotly.express as px
import pandas as pd
import numpy as np
from calendar import month_abbr, month_name
from dash.dependencies import Input, Output

# ======= Initialisation de l'application ======= #
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

#======== Dataframe =========#
df = pd.read_csv("data.csv")
df = df[['CustomerID', 'Gender', 'Location', 'Product_Category', 'Quantity', 'Avg_Price', 'Transaction_Date', 'Month', 'Discount_pct']]

df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

df['Total_price'] = df['Quantity'] * df['Avg_Price'] * (1 - (df['Discount_pct'] / 100)).round(3)

# ========= Statistiques ========= #
def calculer_chiffre_affaire(data):
    return data['Total_price'].sum()

def frequence_meilleure_vente(data, top=10, ascending=False):
    resultat = pd.crosstab(
        [data['Gender'], data['Product_Category']], 
        'Total vente', 
        values=data['Total_price'], 
        aggfunc= lambda x : len(x), 
        rownames=['Sexe', 'Categorie du produit'],
        colnames=['']
    ).reset_index().groupby(
        ['Sexe'], as_index=False, group_keys=True
    ).apply(
        lambda x: x.sort_values('Total vente', ascending=ascending).iloc[:top, :]
    ).reset_index(drop=True).set_index(['Sexe', 'Categorie du produit'])

    return resultat

def indicateur_du_mois(data, current_month = 12, freq=True, abbr=False): 
    previous_month = current_month - 1 if current_month > 1 else 12
    if freq : 
        resultat = data['Month'][(data['Month'] == current_month) | (data['Month'] == previous_month)].value_counts()
        # sort by index
        resultat = resultat.sort_index()
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat
    else:
        resultat = data[(data['Month'] == current_month) | (data['Month'] == previous_month)].groupby('Month').apply(calculer_chiffre_affaire)
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat

def barplot_top_10_ventes(data) :
    df_plot = frequence_meilleure_vente(data, ascending=True)
    graph = px.bar(
        df_plot,
        x='Total vente', 
        y=df_plot.index.get_level_values(1),
        color=df_plot.index.get_level_values(0), 
        barmode='group',
        title="Frequence des 10 meilleures ventes",
        labels={"x": "Fréquence", "y": "Categorie du produit", "color": "Sexe"},
        width=620, height=500
    ).update_layout(
        margin = dict(t=60)
    )
    return graph


def plot_evolution_chiffre_affaire(data):
    df_plot = data.groupby(pd.Grouper(key='Transaction_Date', freq='W')).apply(calculer_chiffre_affaire)[:-1]

    chiffre_evolution = px.line(
        x=df_plot.index, y=df_plot,
        title="Évolution du chiffre d'affaire par semaine",
        labels={"x": "Semaine", "y": "Chiffre d'affaire"},
    ).update_layout(
        width=675,  # largeur
        height=380,  #  hauteur
        margin=dict(t=80, b=20),  # Ajustement des marges
        xaxis=dict(title_font=dict(size=15)),  
        yaxis=dict(title_font=dict(size=15))   
    )
    
    return chiffre_evolution


## Chiffre d'affaire du mois
def plot_chiffre_affaire_mois(data) :
    df_plot = indicateur_du_mois(data, freq=False)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout(
       width=300,  # largeur
        height=250,  # hauteur
        margin=dict(l=20, r=20, t=30, b=10)  # Ajuster les marges
    )
    return indicateur

# Ventes du mois
def plot_vente_mois(data, abbr=False) :
    df_plot = indicateur_du_mois(data, freq=True, abbr=abbr)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout( 
        width=300,  # largeur
        height=250,  #  hauteur
        margin=dict(l=20, r=20, t=30, b=10)  # Ajuster les marges 
    )
    return indicateur


# Fonction du tableau des 100 dernières ventes
def table_100_derniere_ventes(data):
    df_plot_copy = data.copy()
    df_plot_copy['Transaction_Date'] = df_plot_copy['Transaction_Date'].dt.date
    df_plot_copy = df_plot_copy.sort_values('Transaction_Date', ascending=False).head(100)
    
    columns = [
        {"name": "Date", "id": "Transaction_Date"},
        {"name": "Gender", "id": "Gender"},
        {"name": "Location", "id": "Location"},
        {"name": "Product Category", "id": "Product_Category"},
        {"name": "Quantity", "id": "Quantity"},
        {"name": "Avg Price", "id": "Avg_Price"},
        {"name": "Discount (%)", "id": "Discount_pct"},
    ]
    
    return {
        'data': df_plot_copy.to_dict('records'),
        'columns': columns,
    }


# ================ Layout ================== #

app.layout = html.Div([
    # Titre et Dropdown avec possibilité de choisir une zone + option "Choisissez des zones"
    html.Div([
        html.Div(html.H2('ECAP Store', className='text-left'), className='col-md-6', 
                 style={'backgroundColor': 'lightblue', 'padding': '10px'}),
        html.Div(dcc.Dropdown(
            id='dropdown',
            options=[{'label': "Choisissez des zones", 'value': "all"}] +
                    [{'label': loc, 'value': loc} for loc in sorted(df["Location"].dropna().unique())],  
            value="all",  # Valeur par défaut
            clearable=False
        ), className='col-md-6', style={'backgroundColor': 'lightblue', 'padding': '10px'}),
    ], className='row', style={'marginBottom': '20px'}),
    # Contenu principal
    html.Div([
        # Colonne gauche
        html.Div([
            # Les 2 indicateurs
            html.Div([
                html.Div(
                    dcc.Graph(id='indicateur1', style={"height": "100%", "width": "100%"}), 
                    className='col-6 px-1'
                ),
                html.Div(
                    dcc.Graph(id='indicateur2', style={"height": "100%", "width": "100%"}), 
                    className='col-6 px-1'
                ),
            ], className='row mb-3'),

            # Barplot
            html.Div(
                dcc.Graph(id='graph1', style={'height': '300px', 'width': '100%'}),
                className='mb-3'
            )
        ], className='col-12 col-lg-6 px-2'),

        # Colonne droite
        html.Div([
            # Graphique d'évolution du CA
            html.Div(
                dcc.Graph(id='graph2', style={'height': '400px', 'width': '100%'}),
                className='mb-3'
            ),
            
            # Tableau
            html.Div([
                dash_table.DataTable(
                    id='table',
                    page_size=10, 
                    style_table={
                        'width': '100%',
                        'overflowX': 'auto',
                        'margin': '0 auto'
                    },
                    style_cell={
                        'textAlign': 'center', 
                        'padding': '5px',
                        'maxWidth': '100px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'backgroundColor': 'lightgrey'
                    },
                    filter_action="native",
                    sort_action='native',
                    page_action='native',
                    columns=[
                        {"name": "Date", "id": "Transaction_Date"},
                        {"name": "Genre", "id": "Gender"},
                        {"name": "Lieu", "id": "Location"},
                        {"name": "Catégorie", "id": "Product_Category"},
                        {"name": "Quantité", "id": "Quantity"},
                        {"name": "Prix Moyen", "id": "Avg_Price"},
                        {"name": "Remise (%)", "id": "Discount_pct"},
                    ]
                )
            ])
        ], className='col-12 col-lg-6 px-2')

    ], className='row', style={
        'overflowX': 'hidden',
        'width': '100%'
    })
], className='container-fluid', style={
    'maxWidth': '1400px', 
    'margin': '0 auto',
    'overflowX': 'hidden',
    'padding': '15px'
})


# ================= Callbacks ================ #
@app.callback(
    [
        Output('indicateur1', 'figure'),
        Output('indicateur2', 'figure'),
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('table', 'data'),
        Output('table', 'columns')

    ],
    Input('dropdown', 'value')
)
def update_graphs(selected_location):
    # Si "Choisissez des zones" est sélectionné, on prend toutes les locations
    if selected_location == "all":
        filtered_df = df  # Ne pas filtrer
    else:
        filtered_df = df[df["Location"] == selected_location]

    fig_indicateur1 = plot_chiffre_affaire_mois(filtered_df)
    fig_indicateur2 = plot_vente_mois(filtered_df)
    fig_graph1 = barplot_top_10_ventes(filtered_df)
    fig_graph2 = plot_evolution_chiffre_affaire(filtered_df)
    fig_table = table_100_derniere_ventes(filtered_df)

    return fig_indicateur1, fig_indicateur2, fig_graph1, fig_graph2, fig_table['data'], fig_table['columns']


# ================ Run server ================== #
if __name__ == '__main__':
    app.run_serveur(debug=True, port=8073, jupyter_mode="external")


