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

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_risk_segment_dashboard(data):
    # Define high-value customers (top 25% by MonthlyCharges)
    high_value_threshold = data['MonthlyCharges'].quantile(0.75)
    data['CustomerValue'] = data['MonthlyCharges'].apply(
        lambda x: 'High-Value' if x >= high_value_threshold else 'Standard'
    )
    
    # Risk Score Calculation
    data['RiskScore'] = 0
    data['RiskScore'] += (data['Contract'] == 'Month-to-month').astype(int) * 30
    data['RiskScore'] += (data['PaymentMethod'] == 'Electronic check').astype(int) * 20
    data['RiskScore'] += (data['tenure'] < 12).astype(int) * 25
    data['RiskScore'] += (data['InternetService'] == 'Fiber optic').astype(int) * 15
    data['RiskScore'] += (data['OnlineSecurity'] == 'No').astype(int) * 10
    
    # Risk Categories
    def categorize_risk(score):
        if score >= 70:
            return 'Critical Risk'
        elif score >= 50:
            return 'High Risk'
        elif score >= 30:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    data['RiskCategory'] = data['RiskScore'].apply(categorize_risk)
    
    # Demographic segments
    data['DemographicSegment'] = (
        data['SeniorCitizen'].astype(str) + '_' +
        data['Partner'] + '_' +
        data['Dependents']
    )
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'High-Value Customer Risk Matrix',
            'Risk Score Distribution',
            'Churn by Customer Value Segment',
            'Demographic Risk Analysis',
            'Revenue at Risk by Segment',
            'Customer Lifetime Value Analysis'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ],
        row_heights=[0.35, 0.30, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Risk Matrix: MonthlyCharges vs Tenure
    for churn_status in ['No', 'Yes']:
        churn_data = data[data['Churn'] == churn_status]
        fig.add_trace(
            go.Scatter(
                x=churn_data['tenure'],
                y=churn_data['MonthlyCharges'],
                mode='markers',
                name='Retained' if churn_status == 'No' else 'Churned',
                marker=dict(
                    size=8,
                    color='#4CAF50' if churn_status == 'No' else '#FF4444',
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                text=data['CustomerValue'],
                hovertemplate='<b>Tenure: %{x} months</b><br>Monthly: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add quadrants
    median_tenure = data['tenure'].median()
    median_charges = data['MonthlyCharges'].median()
    
    fig.add_hline(y=median_charges, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_vline(x=median_tenure, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Risk Score Distribution
    for churn_status in ['No', 'Yes']:
        risk_data = data[data['Churn'] == churn_status]['RiskScore']
        fig.add_trace(
            go.Histogram(
                x=risk_data,
                name='Retained' if churn_status == 'No' else 'Churned',
                marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )
    
    # Churn by Customer Value
    value_churn = data.groupby(['CustomerValue', 'Churn']).size().unstack(fill_value=0)
    
    for churn_status in ['No', 'Yes']:
        if churn_status in value_churn.columns:
            fig.add_trace(
                go.Bar(
                    x=value_churn.index,
                    y=value_churn[churn_status],
                    name='Retained' if churn_status == 'No' else 'Churned',
                    marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                    text=value_churn[churn_status],
                    textposition='inside'
                ),
                row=2, col=1
            )
    
    # Demographic Risk Analysis
    demo_churn = data.groupby(['SeniorCitizen', 'Partner', 'Dependents', 'Churn']).size().reset_index(name='Count')
    demo_churn['Segment'] = (
        'Senior:' + demo_churn['SeniorCitizen'].astype(str) + 
        ' Partner:' + demo_churn['Partner'] + 
        ' Dep:' + demo_churn['Dependents']
    )
    
    demo_churned = demo_churn[demo_churn['Churn'] == 'Yes'].nlargest(10, 'Count')
    
    fig.add_trace(
        go.Bar(
            y=demo_churned['Segment'],
            x=demo_churned['Count'],
            orientation='h',
            marker_color='#FF4444',
            text=demo_churned['Count'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Churned: %{x}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Revenue at Risk by Risk Category
    risk_category_order = ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']
    revenue_risk = data[data['Churn'] == 'Yes'].groupby('RiskCategory')['TotalCharges'].sum().reindex(risk_category_order, fill_value=0)
    
    colors_risk = ['#8B0000', '#FF4444', '#FFA500', '#FFD700']
    
    fig.add_trace(
        go.Scatter(
            x=revenue_risk.index,
            y=revenue_risk.values,
            mode='lines+markers+text',
            line=dict(color='#FF4444', width=3),
            marker=dict(size=15, color=colors_risk),
            text=[f'${v/1000:.0f}K' for v in revenue_risk.values],
            textposition='top center',
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)',
            hovertemplate='<b>%{x}</b><br>Revenue Lost: $%{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # CLV Analysis: Tenure vs Total Revenue
    clv_data = data.groupby(['tenure', 'Churn']).agg({
        'TotalCharges': 'mean'
    }).reset_index()
    
    for churn_status in ['No', 'Yes']:
        clv_churn = clv_data[clv_data['Churn'] == churn_status]
        fig.add_trace(
            go.Scatter(
                x=clv_churn['tenure'],
                y=clv_churn['TotalCharges'],
                mode='lines',
                name='Retained CLV' if churn_status == 'No' else 'Churned CLV',
                line=dict(width=3),
                marker_color='#4CAF50' if churn_status == 'No' else '#FF4444'
            ),
            row=3, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Tenure (months)", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Charges ($)", row=1, col=1)
    fig.update_xaxes(title_text="Risk Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Customer Value Segment", row=2, col=1)
    fig.update_yaxes(title_text="Customer Count", row=2, col=1)
    fig.update_xaxes(title_text="Churned Customers", row=2, col=2)
    fig.update_xaxes(title_text="Risk Category", row=3, col=1)
    fig.update_yaxes(title_text="Total Revenue Lost ($)", row=3, col=1)
    fig.update_xaxes(title_text="Tenure (months)", row=3, col=2)
    fig.update_yaxes(title_text="Avg Total Charges ($)", row=3, col=2)
    
    fig.update_layout(
        title={
            'text': '<b>Dashboard 4: High-Value Customer Risk Segments</b><br><sub>Identifying & Prioritizing Retention Targets</sub>',
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
            html.H1("High-Value Customer Risk Analysis", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Minimum Monthly Charges:"),
            dcc.Slider(
                id='charges-slider',
                min=df['MonthlyCharges'].min(),
                max=df['MonthlyCharges'].max(),
                value=df['MonthlyCharges'].min(),
                marks={int(i): f'${int(i)}' for i in np.linspace(df['MonthlyCharges'].min(), df['MonthlyCharges'].max(), 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=6),
        
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
            dcc.Graph(id='risk-segment-dashboard')
        ])
    ])
], fluid=True)

@app.callback(
    Output('risk-segment-dashboard', 'figure'),
    [Input('charges-slider', 'value'),
     Input('contract-filter', 'value')]
)
def update_dashboard(min_charges, contract):
    filtered_df = df.copy()
    
    filtered_df = filtered_df[filtered_df['MonthlyCharges'] >= min_charges]
    
    if contract != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    
    return create_risk_segment_dashboard(filtered_df)

if __name__ == '__main__':
    app.run(debug=True, port=8053)
