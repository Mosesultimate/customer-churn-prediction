import pandas as pd
from sqlalchemy import create_engine

## Load cleaned churn dataset
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv")

engine = create_engine("sqlite:///customer_churn.db")

df.to_sql("customers", engine, if_exists="replace", index=False)

print("✅ Data loaded to SQL database successfully.")

