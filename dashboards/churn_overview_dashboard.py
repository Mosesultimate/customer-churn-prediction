import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Load your data
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv')

# Data preprocessing
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Calculate metrics
def calculate_executive_metrics(data):
    total_customers = len(data)
    churned_customers = len(data[data['Churn'] == 'Yes'])
    churn_rate = (churned_customers / total_customers) * 100
    
    churned_revenue = data[data['Churn'] == 'Yes']['TotalCharges'].sum()
    total_revenue = data['TotalCharges'].sum()
    revenue_at_risk_pct = (churned_revenue / total_revenue) * 100
    
    monthly_loss_by_contract = data[data['Churn'] == 'Yes'].groupby('Contract')['MonthlyCharges'].sum().reset_index()
    monthly_loss_by_contract = monthly_loss_by_contract.sort_values('MonthlyCharges', ascending=False)
    
    return {
        'total_customers': total_customers,
        'churned_customers': churned_customers,
        'churn_rate': churn_rate,
        'churned_revenue': churned_revenue,
        'total_revenue': total_revenue,
        'revenue_at_risk_pct': revenue_at_risk_pct,
        'monthly_loss_by_contract': monthly_loss_by_contract
    }

def create_executive_dashboard(data):
    metrics = calculate_executive_metrics(data)
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Overall Churn Rate', 
            'Total Revenue at Risk',
            'Monthly Revenue Loss by Contract Type',
            'Customer Distribution by Contract & Churn',
            'Avg Monthly Charges: Churned vs Retained',
            'Revenue Impact Summary'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'box'}, {'type': 'table'}]
        ],
        row_heights=[0.25, 0.35, 0.40],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Churn Rate KPI
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics['churn_rate'],
            title={'text': f"Churn Rate<br><span style='font-size:0.7em'>({metrics['churned_customers']:,} of {metrics['total_customers']:,} customers)</span>"},
            delta={'reference': 20, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#FF4444'},
                'steps': [
                    {'range': [0, 15], 'color': '#90EE90'},
                    {'range': [15, 25], 'color': '#FFD700'},
                    {'range': [25, 100], 'color': '#FFB6C1'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 25
                }
            },
            number={'suffix': '%', 'font': {'size': 50}}
        ),
        row=1, col=1
    )
    
    # Revenue at Risk KPI
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics['churned_revenue'],
            title={'text': f"Total Revenue Lost<br><span style='font-size:0.7em'>({metrics['revenue_at_risk_pct']:.1f}% of total revenue)</span>"},
            number={'prefix': '$', 'font': {'size': 50, 'color': '#FF4444'}},
            delta={'reference': metrics['total_revenue'] * 0.20, 'relative': False, 'valueformat': '$,.0f'}
        ),
        row=1, col=2
    )
    
    # Monthly Revenue Loss by Contract
    fig.add_trace(
        go.Bar(
            x=metrics['monthly_loss_by_contract']['Contract'],
            y=metrics['monthly_loss_by_contract']['MonthlyCharges'],
            text=metrics['monthly_loss_by_contract']['MonthlyCharges'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            marker=dict(
                color=metrics['monthly_loss_by_contract']['MonthlyCharges'],
                colorscale='Reds',
                showscale=False
            ),
            hovertemplate='<b>%{x}</b><br>Monthly Loss: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Customer Count by Contract and Churn
    contract_churn = data.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    
    for churn_status in ['No', 'Yes']:
        churn_data = contract_churn[contract_churn['Churn'] == churn_status]
        fig.add_trace(
            go.Bar(
                name='Retained' if churn_status == 'No' else 'Churned',
                x=churn_data['Contract'],
                y=churn_data['Count'],
                text=churn_data['Count'],
                textposition='inside',
                marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Box plot of Monthly Charges
    for churn_status in ['No', 'Yes']:
        churn_data = data[data['Churn'] == churn_status]['MonthlyCharges']
        fig.add_trace(
            go.Box(
                y=churn_data,
                name='Retained' if churn_status == 'No' else 'Churned',
                marker_color='#4CAF50' if churn_status == 'No' else '#FF4444',
                boxmean='sd'
            ),
            row=3, col=1
        )
    
    # Summary Table
    summary_data = []
    for contract in data['Contract'].unique():
        contract_df = data[data['Contract'] == contract]
        churned_df = contract_df[contract_df['Churn'] == 'Yes']
        
        summary_data.append([
            contract,
            f"{len(churned_df):,}",
            f"${churned_df['MonthlyCharges'].sum():,.0f}",
            f"${churned_df['TotalCharges'].sum():,.0f}",
            f"{(len(churned_df)/len(contract_df)*100):.1f}%"
        ])
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Contract Type</b>', '<b>Churned<br>Customers</b>', '<b>Monthly<br>Loss</b>', '<b>Total<br>Loss</b>', '<b>Churn<br>Rate</b>'],
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color=[['#f0f0f0', 'white'] * len(summary_data)],
                align='center',
                font=dict(size=11)
            )
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Contract Type", row=2, col=1)
    fig.update_yaxes(title_text="Monthly Revenue Loss ($)", row=2, col=1)
    fig.update_xaxes(title_text="Contract Type", row=2, col=2)
    fig.update_yaxes(title_text="Customer Count", row=2, col=2)
    fig.update_yaxes(title_text="Monthly Charges ($)", row=3, col=1)
    
    fig.update_layout(
        title={
            'text': '<b>Dashboard 1: Executive Churn Overview</b><br><sub>Business Impact Analysis</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1200,
        showlegend=True,
        legend=dict(x=0.85, y=0.55),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Customer Churn Analytics", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Contract Type:"),
            dcc.Dropdown(
                id='contract-filter',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': c, 'value': c} for c in df['Contract'].unique()],
                value='All',
                clearable=False
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Filter by Internet Service:"),
            dcc.Dropdown(
                id='internet-filter',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': i, 'value': i} for i in df['InternetService'].unique()],
                value='All',
                clearable=False
            )
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='executive-dashboard')
        ])
    ])
], fluid=True)

@app.callback(
    Output('executive-dashboard', 'figure'),
    [Input('contract-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_dashboard(contract, internet):
    filtered_df = df.copy()
    
    if contract != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    
    if internet != 'All':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    return create_executive_dashboard(filtered_df)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
    