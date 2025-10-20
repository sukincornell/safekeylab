#!/usr/bin/env python3
"""
Generate secure API keys for Aegis clients
"""

import secrets
import string
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_api_key(prefix: str = "sk_live_", length: int = 32) -> str:
    """Generate a secure API key"""
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{key}"

def hash_api_key(api_key: str) -> str:
    """Generate SHA256 hash of API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def create_api_key_record(
    name: str,
    scopes: list = None,
    rate_limit: int = 10000,
    expires_days: int = 365
) -> Dict:
    """Create a complete API key record"""

    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    if scopes is None:
        scopes = ["detect", "anonymize", "process", "batch"]

    record = {
        "id": secrets.token_urlsafe(16),
        "name": name,
        "key_hash": key_hash,
        "key_prefix": api_key[:12] + "...",  # Store prefix for identification
        "scopes": scopes,
        "rate_limit": rate_limit,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat(),
        "last_used": None,
        "status": "active",
        "metadata": {
            "environment": "production",
            "version": "1.0.0"
        }
    }

    return {
        "api_key": api_key,  # Return plaintext only once
        "record": record
    }

def save_key_to_database(record: Dict, db_path: str = "./keys.json"):
    """Save API key record to database (JSON file for demo)"""
    try:
        with open(db_path, 'r') as f:
            keys = json.load(f)
    except FileNotFoundError:
        keys = []

    keys.append(record)

    with open(db_path, 'w') as f:
        json.dump(keys, f, indent=2)

    return True

def main():
    parser = argparse.ArgumentParser(description='Generate Aegis API Keys')
    parser.add_argument('--name', required=True, help='Client/Application name')
    parser.add_argument('--scopes', nargs='+', help='API scopes (default: all)')
    parser.add_argument('--rate-limit', type=int, default=10000, help='Requests per minute')
    parser.add_argument('--expires-days', type=int, default=365, help='Days until expiration')
    parser.add_argument('--save', action='store_true', help='Save to database')

    args = parser.parse_args()

    # Create API key
    result = create_api_key_record(
        name=args.name,
        scopes=args.scopes,
        rate_limit=args.rate_limit,
        expires_days=args.expires_days
    )

    # Display results
    print("\n" + "="*60)
    print("🔑 AEGIS API KEY GENERATED")
    print("="*60)
    print(f"\n📌 Client: {args.name}")
    print(f"🔐 API Key: {result['api_key']}")
    print(f"\n⚠️  IMPORTANT: Save this key securely!")
    print("    This is the only time you'll see the full key.")
    print("\n" + "-"*60)
    print(f"📊 Rate Limit: {args.rate_limit} requests/minute")
    print(f"🔓 Scopes: {', '.join(result['record']['scopes'])}")
    print(f"📅 Expires: {result['record']['expires_at']}")
    print(f"🆔 Key ID: {result['record']['id']}")

    # Save to database if requested
    if args.save:
        if save_key_to_database(result['record']):
            print(f"\n✅ Key record saved to database")

    print("\n" + "="*60)

    # Output for environment variable
    print("\n📋 Add to your .env or environment:")
    print(f"AEGIS_API_KEY={result['api_key']}")
    print("\n")

if __name__ == "__main__":
    main()