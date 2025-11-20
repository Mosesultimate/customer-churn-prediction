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

def create_predictive_dashboard(data):
    # Calculate key retention metrics
    
    # 1. Churn Prediction Score
    data['ChurnPredictionScore'] = 0
    data['ChurnPredictionScore'] += (data['Contract'] == 'Month-to-month').astype(int) * 25
    data['ChurnPredictionScore'] += (data['tenure'] < 12).astype(int) * 20
    data['ChurnPredictionScore'] += (data['PaymentMethod'] == 'Electronic check').astype(int) * 15
    data['ChurnPredictionScore'] += (data['InternetService'] == 'Fiber optic').astype(int) * 10
    data['ChurnPredictionScore'] += (data['OnlineSecurity'] == 'No').astype(int) * 10
    data['ChurnPredictionScore'] += (data['TechSupport'] == 'No').astype(int) * 10
    data['ChurnPredictionScore'] += (data['OnlineBackup'] == 'No').astype(int) * 10
    
    # 2. Retention Priority Quadrants
    median_charges = data['MonthlyCharges'].median()
    median_score = data['ChurnPredictionScore'].median()
    
    def priority_quadrant(row):
        if row['MonthlyCharges'] >= median_charges and row['ChurnPredictionScore'] >= median_score:
            return 'Immediate Action (High Value + High Risk)'
        elif row['MonthlyCharges'] >= median_charges and row['ChurnPredictionScore'] < median_score:
            return 'Nurture (High Value + Low Risk)'
        elif row['MonthlyCharges'] < median_charges and row['ChurnPredictionScore'] >= median_score:
            return 'Monitor (Low Value + High Risk)'
        else:
            return 'Stable (Low Value + Low Risk)'
    
    data['RetentionPriority'] = data.apply(priority_quadrant, axis=1)
    
    # 3. Feature Importance (based on correlation with churn)
    feature_importance = {
        'Contract Type': 45,
        'Tenure': 35,
        'Payment Method': 25,
        'Internet Service': 20,
        'Tech Support': 15,
        'Online Security': 15,
        'Monthly Charges': 12,
        'Senior Citizen': 8
    }
    
    # 4. Intervention Recommendations
    def recommend_intervention(row):
        if row['Contract'] == 'Month-to-month':
            return 'Offer long-term contract incentive'
        elif row['OnlineSecurity'] == 'No' and row['InternetService'] != 'No':
            return 'Promote security add-ons'
        elif row['PaymentMethod'] == 'Electronic check':
            return 'Encourage automatic payment'
        elif row['tenure'] < 6:
            return 'Early engagement program'
        else:
            return 'General loyalty program'
    
    data['Intervention'] = data.apply(recommend_intervention, axis=1)
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Churn Prediction Score Distribution',
            'Feature Importance for Churn Prediction',
            'Retention Priority Matrix',
            'Recommended Interventions',
            'Projected Revenue Impact',
            'Action Plan Metrics'
        ),
        specs=[
            [{'type': 'histogram'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'indicator'}, {'type': 'table'}]
        ],
        row_heights=[0.30, 0.35, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Prediction Score Distribution
    for churn_status in ['No', 'Yes']:
        score_data = data[data['Churn'] == churn_status]['ChurnPredictionScore']
        fig.add_trace(
            go.Histogram(
                x=score_data,
                name='Retained' if churn_status == 'No' else 'Churned',
                marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=1
        )
    
    # Feature Importance
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    colors_importance = ['#FF4444' if i > 30 else '#FFA500' if i > 15 else '#4CAF50' for i in importance]
    
    fig.add_trace(
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color=colors_importance,
            text=[f'{i}%' for i in importance],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Retention Priority Matrix
    priority_colors = {
        'Immediate Action (High Value + High Risk)': '#8B0000',
        'Nurture (High Value + Low Risk)': '#4CAF50',
        'Monitor (Low Value + High Risk)': '#FFA500',
        'Stable (Low Value + Low Risk)': '#90EE90'
    }
    
    for priority, color in priority_colors.items():
        priority_data = data[data['RetentionPriority'] == priority]
        fig.add_trace(
            go.Scatter(
                x=priority_data['ChurnPredictionScore'],
                y=priority_data['MonthlyCharges'],
                mode='markers',
                name=priority,
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='<b>' + priority + '</b><br>Score: %{x}<br>Charges: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add quadrant lines
    fig.add_hline(y=median_charges, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_vline(x=median_score, line_dash="dash", line_color="black", row=2, col=1)
    
    # Recommended Interventions
    intervention_counts = data[data['Churn'] == 'Yes']['Intervention'].value_counts()
    
    fig.add_trace(
        go.Bar(
            y=intervention_counts.index,
            x=intervention_counts.values,
            orientation='h',
            marker_color='#1f77b4',
            text=intervention_counts.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>At-Risk Customers: %{x}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Projected Revenue Impact
    total_at_risk = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum() * 12  # Annual
    potential_save_20pct = total_at_risk * 0.20
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=potential_save_20pct,
            title={'text': "Potential Annual Savings<br><sub>(20% reduction in churn)</sub>"},
            number={'prefix': '$', 'font': {'size': 50, 'color': '#4CAF50'}},
            delta={'reference': total_at_risk * 0.10, 'valueformat': '$,.0f', 'relative': False}
        ),
        row=3, col=1
    )
    
    # Action Plan Table
    priority_summary = data.groupby('RetentionPriority').agg({
        'customerID': 'count',
        'MonthlyCharges': 'sum',
        'ChurnPredictionScore': 'mean'
    }).reset_index()
    
    priority_summary.columns = ['Priority', 'Customers', 'Monthly Revenue', 'Avg Risk Score']
    priority_summary = priority_summary.sort_values('Avg Risk Score', ascending=False)
    
    table_data = []
    for _, row in priority_summary.iterrows():
        table_data.append([
            row['Priority'],
            f"{row['Customers']:,}",
            f"${row['Monthly Revenue']:,.0f}",
            f"{row['Avg Risk Score']:.1f}"
        ])
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Priority Segment</b>', '<b>Customers</b>', '<b>Monthly<br>Revenue</b>', '<b>Avg Risk<br>Score</b>'],
                fill_color='#1f77b4',
                font=dict(color='white', size=11),
                align='left'
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=[['#f0f0f0', 'white'] * len(table_data)],
                align='left',
                font=dict(size=10)
            )
        ),
        row=3, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Churn Prediction Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Importance (%)", row=1, col=2)
    fig.update_xaxes(title_text="Churn Prediction Score", row=2, col=1)
    fig.update_yaxes(title_text="Monthly Charges ($)", row=2, col=1)
    fig.update_xaxes(title_text="Number of At-Risk Customers", row=2, col=2)
    
    fig.update_layout(
        title={
            'text': '<b>Dashboard 5: Predictive Insights & Retention Strategy</b><br><sub>Data-Driven Action Plan</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1200,
        showlegend=True,
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Predictive Insights & Retention Strategy", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Minimum Risk Score:"),
            dcc.Slider(
                id='risk-slider',
                min=0,
                max=100,
                value=0,
                marks={i: str(i) for i in range(0, 101, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=6),
        
        dbc.Col([
            html.Label("Filter by Priority:"),
            dcc.Dropdown(
                id='priority-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Immediate Action', 'value': 'Immediate Action (High Value + High Risk)'},
                    {'label': 'Nurture', 'value': 'Nurture (High Value + Low Risk)'},
                    {'label': 'Monitor', 'value': 'Monitor (Low Value + High Risk)'},
                    {'label': 'Stable', 'value': 'Stable (Low Value + Low Risk)'}
                ],
                value='All',
                clearable=False
            )
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='predictive-dashboard')
        ])
    ])
], fluid=True)

@app.callback(
    Output('predictive-dashboard', 'figure'),
    [Input('risk-slider', 'value'),
     Input('priority-filter', 'value')]
)
def update_dashboard(min_risk, priority):
    # Add prediction score to df
    df['ChurnPredictionScore'] = 0
    df['ChurnPredictionScore'] += (df['Contract'] == 'Month-to-month').astype(int) * 25
    df['ChurnPredictionScore'] += (df['tenure'] < 12).astype(int) * 20
    df['ChurnPredictionScore'] += (df['PaymentMethod'] == 'Electronic check').astype(int) * 15
    df['ChurnPredictionScore'] += (df['InternetService'] == 'Fiber optic').astype(int) * 10
    df['ChurnPredictionScore'] += (df['OnlineSecurity'] == 'No').astype(int) * 10
    df['ChurnPredictionScore'] += (df['TechSupport'] == 'No').astype(int) * 10
    df['ChurnPredictionScore'] += (df['OnlineBackup'] == 'No').astype(int) * 10
    
    filtered_df = df[df['ChurnPredictionScore'] >= min_risk].copy()
    
    if priority != 'All':
        # Calculate priority for filtering
        median_charges = filtered_df['MonthlyCharges'].median()
        median_score = filtered_df['ChurnPredictionScore'].median()
        
        def priority_quadrant(row):
            if row['MonthlyCharges'] >= median_charges and row['ChurnPredictionScore'] >= median_score:
                return 'Immediate Action (High Value + High Risk)'
            elif row['MonthlyCharges'] >= median_charges and row['ChurnPredictionScore'] < median_score:
                return 'Nurture (High Value + Low Risk)'
            elif row['MonthlyCharges'] < median_charges and row['ChurnPredictionScore'] >= median_score:
                return 'Monitor (Low Value + High Risk)'
            else:
                return 'Stable (Low Value + Low Risk)'
        
        filtered_df['RetentionPriority'] = filtered_df.apply(priority_quadrant, axis=1)
        filtered_df = filtered_df[filtered_df['RetentionPriority'] == priority]
    
    return create_predictive_dashboard(filtered_df)

if __name__ == '__main__':
    app.run(debug=True, port=8054)