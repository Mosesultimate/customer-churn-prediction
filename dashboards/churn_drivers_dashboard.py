import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np

# Load your data
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv')

# Data preprocessing
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Tenure categorization
def categorize_tenure(tenure):
    if tenure <= 6:
        return '0-6 months'
    elif tenure <= 12:
        return '7-12 months'
    elif tenure <= 24:
        return '13-24 months'
    elif tenure <= 36:
        return '25-36 months'
    elif tenure <= 48:
        return '37-48 months'
    else:
        return '48+ months'

df['TenureCohort'] = df['tenure'].apply(categorize_tenure)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_tenure_payment_dashboard(data):
    cohort_order = ['0-6 months', '7-12 months', '13-24 months', '25-36 months', '37-48 months', '48+ months']
    
    # Churn rate by tenure cohort
    tenure_churn = data.groupby(['TenureCohort', 'Churn']).size().unstack(fill_value=0)
    tenure_churn['Total'] = tenure_churn.sum(axis=1)
    tenure_churn['ChurnRate'] = (tenure_churn.get('Yes', 0) / tenure_churn['Total']) * 100
    tenure_churn = tenure_churn.reindex(cohort_order)
    
    # Churn by Payment Method
    payment_churn = data.groupby(['PaymentMethod', 'Churn']).size().unstack(fill_value=0)
    payment_churn['Total'] = payment_churn.sum(axis=1)
    payment_churn['ChurnRate'] = (payment_churn.get('Yes', 0) / payment_churn['Total']) * 100
    payment_churn = payment_churn.sort_values('ChurnRate', ascending=False)
    
    # Monthly tenure analysis (first 24 months)
    monthly_tenure = data[data['tenure'] <= 24].copy()
    monthly_churn_rate = monthly_tenure.groupby('tenure').apply(
        lambda x: (len(x[x['Churn'] == 'Yes']) / len(x) * 100) if len(x) > 0 else 0
    ).reset_index(name='ChurnRate')
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Churn Rate by Tenure Cohort',
            'Customer Count by Tenure Cohort',
            'Detailed: First 24 Months Churn Rate',
            'Churn Rate by Payment Method',
            'Payment Method Distribution (Churned vs Retained)',
            'Tenure-Payment Method Interaction'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ],
        row_heights=[0.30, 0.35, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Churn Rate by Tenure Cohort
    colors = ['#FF4444' if rate > 40 else '#FFA500' if rate > 25 else '#4CAF50' 
              for rate in tenure_churn['ChurnRate']]
    
    fig.add_trace(
        go.Bar(
            x=tenure_churn.index,
            y=tenure_churn['ChurnRate'],
            text=tenure_churn['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>',
            name='Churn Rate'
        ),
        row=1, col=1
    )
    
    # Customer Count by Tenure
    fig.add_trace(
        go.Bar(
            x=tenure_churn.index,
            y=tenure_churn.get('No', 0),
            name='Retained',
            marker_color='#4CAF50',
            text=tenure_churn.get('No', 0),
            textposition='inside'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=tenure_churn.index,
            y=tenure_churn.get('Yes', 0),
            name='Churned',
            marker_color='#FF4444',
            text=tenure_churn.get('Yes', 0),
            textposition='inside'
        ),
        row=1, col=2
    )
    
    # Detailed First 24 Months
    fig.add_trace(
        go.Scatter(
            x=monthly_churn_rate['tenure'],
            y=monthly_churn_rate['ChurnRate'],
            mode='lines+markers',
            line=dict(color='#FF4444', width=3),
            marker=dict(size=8, color='#FF4444'),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)',
            hovertemplate='<b>Month %{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>',
            name='Monthly Churn Rate'
        ),
        row=2, col=1
    )
    
    # Payment Method Churn Rate
    payment_colors = ['#FF4444' if rate > 35 else '#FFA500' if rate > 25 else '#4CAF50' 
                      for rate in payment_churn['ChurnRate']]
    
    fig.add_trace(
        go.Bar(
            x=payment_churn['ChurnRate'],
            y=payment_churn.index,
            orientation='h',
            text=payment_churn['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            marker=dict(color=payment_colors),
            name='Churn Rate by Payment'
        ),
        row=2, col=2
    )
    
    # Payment Method Distribution
    payment_dist = data.groupby(['PaymentMethod', 'Churn']).size().unstack(fill_value=0)
    
    for churn_status in ['No', 'Yes']:
        if churn_status in payment_dist.columns:
            fig.add_trace(
                go.Bar(
                    x=payment_dist.index,
                    y=payment_dist[churn_status],
                    name='Retained' if churn_status == 'No' else 'Churned',
                    marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                    text=payment_dist[churn_status],
                    textposition='inside'
                ),
                row=3, col=1
            )
    
    # Tenure-Payment Interaction
    tenure_payment = data.groupby(['TenureCohort', 'PaymentMethod', 'Churn']).size().reset_index(name='Count')
    tenure_payment_churn = tenure_payment[tenure_payment['Churn'] == 'Yes'].copy()
    
    tenure_payment_churn['TenureNum'] = tenure_payment_churn['TenureCohort'].map(
        {cohort: i for i, cohort in enumerate(cohort_order)}
    )
    tenure_payment_churn['PaymentNum'] = tenure_payment_churn['PaymentMethod'].astype('category').cat.codes
    
    fig.add_trace(
        go.Scatter(
            x=tenure_payment_churn['TenureNum'],
            y=tenure_payment_churn['PaymentNum'],
            mode='markers',
            marker=dict(
                size=tenure_payment_churn['Count'],
                sizemode='area',
                sizeref=2.*max(tenure_payment_churn['Count'])/(40.**2),
                color=tenure_payment_churn['Count'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Churned<br>Count", x=1.15)
            ),
            text=tenure_payment_churn['Count'],
            hovertemplate='<b>%{text} churned customers</b><extra></extra>',
            name='Churn Intensity'
        ),
        row=3, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Tenure Cohort", row=1, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Months Since Signup", row=2, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Churn Rate (%)", row=2, col=2)
    fig.update_xaxes(
        title_text="Tenure Cohort", 
        ticktext=cohort_order,
        tickvals=list(range(len(cohort_order))),
        row=3, col=2
    )
    fig.update_yaxes(
        title_text="Payment Method",
        ticktext=payment_churn.index.tolist(),
        tickvals=list(range(len(payment_churn))),
        row=3, col=2
    )
    
    fig.update_layout(
        title={
            'text': '<b>Dashboard 2: Tenure & Payment Analysis</b><br><sub>Understanding Early Churn & Payment Risk</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1200,
        showlegend=True,
        barmode='stack',
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Tenure & Payment Analysis", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Senior Citizen:"),
            dcc.Dropdown(
                id='senior-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'}
                ],
                value='All',
                clearable=False
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Filter by Contract:"),
            dcc.Dropdown(
                id='contract-filter',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': c, 'value': c} for c in df['Contract'].unique()],
                value='All',
                clearable=False
            )
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='tenure-payment-dashboard')
        ])
    ])
], fluid=True)

@app.callback(
    Output('tenure-payment-dashboard', 'figure'),
    [Input('senior-filter', 'value'),
     Input('contract-filter', 'value')]
)
def update_dashboard(senior, contract):
    filtered_df = df.copy()
    
    if senior != 'All':
        filtered_df = filtered_df[filtered_df['SeniorCitizen'].astype(str) == senior]
    
    if contract != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    
    return create_tenure_payment_dashboard(filtered_df)

if __name__ == '__main__':
    app.run(debug=True, port=8051)
    