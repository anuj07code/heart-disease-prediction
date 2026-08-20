"""
HeartGuard AI — Database Initialization
========================================
Creates all tables and optionally seeds demo accounts.

Usage:
    python init_db.py
"""
import os
import sys
import sqlite3

# Add backend to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'backend'))

from app import app
from models import db, User


def ensure_user_columns():
    """Ensure new User model columns exist in the SQLite database."""
    db_path = os.path.join(PROJECT_DIR, 'instance', 'database.db')
    if not os.path.exists(db_path):
        return

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(user)")
        existing = {row[1] for row in cur.fetchall()}

        migrations = []
        if 'email_confirmed' not in existing:
            migrations.append("ALTER TABLE user ADD COLUMN email_confirmed INTEGER DEFAULT 0")
        if 'reset_token' not in existing:
            migrations.append("ALTER TABLE user ADD COLUMN reset_token TEXT")
        if 'reset_expires' not in existing:
            migrations.append("ALTER TABLE user ADD COLUMN reset_expires TEXT")

        for sql in migrations:
            cur.execute(sql)
            print(f"[OK] Added column to user table: {sql}")
        if migrations:
            conn.commit()


def init_database():
    """Create database tables."""

    os.makedirs(os.path.join(PROJECT_DIR, 'instance'), exist_ok=True)

    with app.app_context():
        # Create all tables
        db.create_all()
        print("[OK] Database tables created.")

        ensure_user_columns()

        # Seed demo accounts for testing
        if User.query.count() == 0:
            print("[OK] Seeding demo accounts...")

            # Demo Doctor
            doctor = User(
                name="Dr. Sarah Johnson",
                email="doctor@heartguard.ai",
                role="doctor",
                email_confirmed=True
            )
            doctor.set_password("demo123")
            db.session.add(doctor)

            # Demo Patient
            patient = User(
                name="John Smith",
                email="patient@heartguard.ai",
                role="patient",
                email_confirmed=True
            )
            patient.set_password("demo123")
            db.session.add(patient)

            db.session.commit()
            print("[OK] Demo accounts created:")
            print("     Doctor: doctor@heartguard.ai / demo123")
            print("     Patient: patient@heartguard.ai / demo123")

        print(f"[OK] Database location: {app.config['SQLALCHEMY_DATABASE_URI']}")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  HeartGuard AI — Database Setup")
    print("=" * 50)
    init_database()
    print("\nDone! You can now run: python run.py\n")
