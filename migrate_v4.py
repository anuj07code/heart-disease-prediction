"""
HeartGuard V4 Database Migration
Adds missing V4 columns to existing database safely.
"""
import sqlite3
import os
import sys

# Determine the database path
INSTANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
DB_PATH = os.path.join(INSTANCE_DIR, 'database.db')

if not os.path.exists(DB_PATH):
    print(f"[!] Database not found at {DB_PATH}")
    print("    Run the app once to create it, then run this migration.")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def column_exists(table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns

# ── User table migrations ──
user_columns = {
    'caregiver_id': 'INTEGER',
    'heart_points': 'INTEGER DEFAULT 0',
    'language_preference': "VARCHAR(10) DEFAULT 'en'",
}

for col, col_type in user_columns.items():
    if not column_exists('user', col):
        sql = f"ALTER TABLE user ADD COLUMN {col} {col_type}"
        cursor.execute(sql)
        print(f"  [+] Added user.{col}")
    else:
        print(f"  [=] user.{col} already exists")

# ── Report table migrations ──
report_columns = {
    'tobacco': 'INTEGER',
    'obesity': 'INTEGER',
    'exercise_freq': 'INTEGER',
    'unhealthy_diet': 'INTEGER',
    'stress_level': 'INTEGER',
    'genetics': 'INTEGER',
    'bmi': 'REAL',
    'ai_health_tips': 'TEXT',
    'doctor_note': 'TEXT',
    'verified_at': 'DATETIME',
}

for col, col_type in report_columns.items():
    if not column_exists('report', col):
        sql = f"ALTER TABLE report ADD COLUMN {col} {col_type}"
        cursor.execute(sql)
        print(f"  [+] Added report.{col}")
    else:
        print(f"  [=] report.{col} already exists")

conn.commit()
conn.close()

print("\n✅ Migration complete! You can now run the app with: python run.py")
