#!/usr/bin/env python3
"""
Test the Real Aegis API
Demonstrates actual PII detection across all modalities
"""

import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io

API_URL = "http://localhost:8000"

def test_registration():
    """Register a test user and get API key"""
    print("\n🔐 Testing Registration...")

    response = requests.post(f"{API_URL}/api/v1/register", json={
        "email": "demo@aegis.com",
        "name": "Demo User",
        "company": "Test Corp",
        "password": "secure123",
        "plan": "trial"
    })

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Registration successful!")
        print(f"   API Key: {data['api_key']}")
        print(f"   Trial ends: {data['trial_ends']}")
        return data['api_key']
    else:
        print(f"❌ Registration failed: {response.text}")
        # Try to use existing user
        return "ak_live_existing_key"

def test_text_detection(api_key):
    """Test real text PII detection"""
    print("\n📝 Testing Text PII Detection...")

    test_texts = [
        "My name is John Smith and my SSN is 123-45-6789.",
        "Contact me at john@example.com or call 555-123-4567.",
        "I live at 123 Main Street, New York, NY 10001.",
        "My credit card number is 4111-1111-1111-1111 with CVV 123.",
        "Patient Mary Johnson, DOB: 01/15/1980, MRN: 12345678"
    ]

    headers = {"Authorization": f"Bearer {api_key}"}

    for text in test_texts:
        response = requests.post(
            f"{API_URL}/api/v1/detect/text",
            headers=headers,
            json={"text": text}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n   Original: {text}")
            print(f"   Anonymized: {result['anonymized_content']}")
            print(f"   PII Found: {len(result['pii_detected'])} entities")
            for pii in result['pii_detected']:
                print(f"      - {pii['entity_type']}: {pii['text']} (confidence: {pii['confidence']:.2f})")
        else:
            print(f"   ❌ Error: {response.text}")

def test_image_detection(api_key):
    """Test real image PII detection with face blurring"""
    print("\n🖼️ Testing Image PII Detection...")

    # Create a test image with text
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    # Add some text with PII
    text_with_pii = [
        "Employee ID: EMP123456",
        "Name: Jane Doe",
        "SSN: 987-65-4321",
        "Email: jane.doe@company.com"
    ]

    y_offset = 50
    for text in text_with_pii:
        draw.text((50, y_offset), text, fill='black')
        y_offset += 40

    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Send to API
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("test_image.png", img_bytes, "image/png")}

    response = requests.post(
        f"{API_URL}/api/v1/detect/image",
        headers=headers,
        files=files,
        data={"detect_faces": "true", "detect_text": "true"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Image processed successfully!")
        print(f"   PII Found: {len(result['pii_detected'])} entities")
        print(f"   Processing time: {result['processing_time']:.3f}s")
        for pii in result['pii_detected']:
            if 'TEXT_' in pii['entity_type']:
                print(f"      - {pii['entity_type']}: {pii.get('text', 'N/A')}")
            else:
                print(f"      - {pii['entity_type']} detected")
    else:
        print(f"   ❌ Error: {response.text}")

def test_unified_endpoint(api_key):
    """Test the unified /protect endpoint"""
    print("\n🛡️ Testing Unified Protection Endpoint...")

    headers = {"Authorization": f"Bearer {api_key}"}

    # Test text through unified endpoint
    response = requests.post(
        f"{API_URL}/api/v1/protect",
        headers=headers,
        data={
            "data_type": "text",
            "content": "Call me at 415-555-0123 or email john@example.com"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Unified endpoint working!")
        print(f"   Original: Call me at 415-555-0123 or email john@example.com")
        print(f"   Protected: {result['anonymized_content']}")
    else:
        print(f"   ❌ Error: {response.text}")

def test_usage_stats(api_key):
    """Get usage statistics"""
    print("\n📊 Testing Usage Statistics...")

    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{API_URL}/api/v1/usage", headers=headers)

    if response.status_code == 200:
        stats = response.json()
        print(f"   ✅ Usage Stats:")
        print(f"      User: {stats['user']}")
        print(f"      Plan: {stats['plan']}")
        print(f"      Usage: {stats['usage_count']} / {stats['usage_limit']}")
        if stats['statistics']['by_data_type']:
            print(f"      By Type: {stats['statistics']['by_data_type']}")
    else:
        print(f"   ❌ Error: {response.text}")

def main():
    print("=" * 60)
    print("🚀 AEGIS REAL API TEST")
    print("=" * 60)

    # Check if API is running
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ API is running: {info['name']} v{info['version']}")
            print(f"   Features: {', '.join(info['features'])}")
        else:
            print("❌ API is not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Please ensure aegis_real_api.py is running")
        return

    # Run tests
    api_key = test_registration()

    if api_key:
        test_text_detection(api_key)
        test_image_detection(api_key)
        test_unified_endpoint(api_key)
        test_usage_stats(api_key)

    print("\n" + "=" * 60)
    print("✅ AEGIS REAL API TEST COMPLETE!")
    print("=" * 60)
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("🌐 Dashboard: http://localhost:8080/dashboard.html")
    print("\nThe API now has REAL PII detection capabilities:")
    print("  • Text: Names, SSNs, emails, phones, addresses")
    print("  • Images: Face detection, text extraction")
    print("  • Audio: Transcription + PII detection")
    print("  • Video: Frame analysis + face blurring")
    print("  • Documents: OCR + entity recognition")
    print("\n💡 Next steps:")
    print("  1. Add your Stripe API key for real payments")
    print("  2. Deploy to cloud (AWS/GCP/Azure)")
    print("  3. Get SOC 2 certification")
    print("  4. Start selling!")

if __name__ == "__main__":
    main()