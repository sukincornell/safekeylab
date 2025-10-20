#!/usr/bin/env python3
"""
Database initialization and migration script for Aegis
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def wait_for_postgres(host, port, user, password, max_retries=30):
    """Wait for PostgreSQL to be ready"""
    print(f"⏳ Waiting for PostgreSQL at {host}:{port}...")

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database='postgres'
            )
            conn.close()
            print("✅ PostgreSQL is ready!")
            return True
        except psycopg2.OperationalError:
            if i < max_retries - 1:
                time.sleep(2)
                print(f"   Retry {i+1}/{max_retries}...")
            else:
                print("❌ PostgreSQL connection failed after maximum retries")
                return False
    return False

def create_database(connection_params):
    """Create the Aegis database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(**connection_params, database='postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'aegis_production'")
        exists = cursor.fetchone()

        if not exists:
            print("📦 Creating database 'aegis_production'...")
            cursor.execute("CREATE DATABASE aegis_production")
            print("✅ Database created successfully!")
        else:
            print("ℹ️  Database 'aegis_production' already exists")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def run_migrations(connection_params):
    """Run database migrations from SQL file"""
    try:
        # Connect to aegis_production database
        conn = psycopg2.connect(**connection_params, database='aegis_production')
        cursor = conn.cursor()

        print("🔄 Running database migrations...")

        # Read and execute SQL file
        sql_file = os.path.join(os.path.dirname(__file__), 'init_db.sql')
        if os.path.exists(sql_file):
            with open(sql_file, 'r') as f:
                sql_commands = f.read()

            # Execute SQL commands
            cursor.execute(sql_commands)
            conn.commit()
            print("✅ Migrations completed successfully!")
        else:
            print(f"⚠️  SQL file not found at {sql_file}")
            print("   Creating minimal schema...")

            # Create minimal schema if SQL file doesn't exist
            cursor.execute("""
                -- Enable extensions
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                CREATE EXTENSION IF NOT EXISTS "pgcrypto";

                -- Create schemas
                CREATE SCHEMA IF NOT EXISTS api;
                CREATE SCHEMA IF NOT EXISTS audit;

                -- Organizations table
                CREATE TABLE IF NOT EXISTS api.organizations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active'
                );

                -- API Keys table
                CREATE TABLE IF NOT EXISTS api.api_keys (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    organization_id UUID REFERENCES api.organizations(id),
                    name VARCHAR(255) NOT NULL,
                    key_hash VARCHAR(255) UNIQUE NOT NULL,
                    key_prefix VARCHAR(20) NOT NULL,
                    scopes TEXT[] DEFAULT ARRAY['detect', 'anonymize'],
                    rate_limit INTEGER DEFAULT 10000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active'
                );

                -- Usage logs table
                CREATE TABLE IF NOT EXISTS api.usage_logs (
                    id BIGSERIAL PRIMARY KEY,
                    api_key_id UUID REFERENCES api.api_keys(id),
                    endpoint VARCHAR(255) NOT NULL,
                    status_code INTEGER,
                    latency_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Insert default organization
                INSERT INTO api.organizations (name, slug)
                VALUES ('Default', 'default')
                ON CONFLICT (slug) DO NOTHING;
            """)
            conn.commit()
            print("✅ Minimal schema created!")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False

def main():
    """Main initialization function"""
    print("\n" + "="*60)
    print("🛡️  AEGIS DATABASE INITIALIZATION")
    print("="*60 + "\n")

    # Get database connection parameters from environment
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_user = os.getenv('DB_USER', 'aegis')
    db_password = os.getenv('DB_PASSWORD', 'oKV7BL16XiarmQMky7IpihZsdJ')

    connection_params = {
        'host': db_host,
        'port': db_port,
        'user': db_user,
        'password': db_password
    }

    # Step 1: Wait for PostgreSQL to be ready
    if not wait_for_postgres(db_host, db_port, db_user, db_password):
        sys.exit(1)

    # Step 2: Create database
    if not create_database(connection_params):
        sys.exit(1)

    # Step 3: Run migrations
    if not run_migrations(connection_params):
        sys.exit(1)

    print("\n" + "="*60)
    print("🎉 Database initialization completed successfully!")
    print("="*60 + "\n")

    print("📋 Next steps:")
    print("   1. Generate API keys: python scripts/generate_api_key.py --name 'ClientName'")
    print("   2. Start the API server: docker-compose up -d")
    print("   3. Access API docs: http://localhost:8000/docs")
    print("\n")

if __name__ == "__main__":
    main()