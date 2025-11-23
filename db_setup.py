import sqlite3

conn = sqlite3.connect('customer_churn.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER,
    partner TEXT,
    dependents TEXT,
    tenure INTEGER,
    monthly_charges REAL,
    total_charges REAL,
    churn TEXT
)
''')

conn.commit()
conn.close()

print("✅ Database and table created successfully.")