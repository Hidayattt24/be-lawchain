#!/usr/bin/env python3
"""
Test script untuk LawChain Backend API
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_health():
    """Test health endpoint"""
    print("🩺 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return False

def test_system_info():
    """Test system info endpoint"""
    print("\n📊 Testing system info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False

def test_ask_endpoint_langchain():
    """Test ask endpoint with LangChain method"""
    print("\n🦜 Testing ask endpoint with LangChain method...")
    print("⏳ This may take several minutes for local LLM processing...")
    payload = {
        "question": "Apa yang dimaksud dengan Pancasila?",
        "method": "langchain"
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout for local LLM
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ LangChain ask test failed: {e}")
        return False

def test_ask_endpoint_native():
    """Test ask endpoint with Native method"""
    print("\n🔧 Testing ask endpoint with Native method...")
    print("⏳ This may take several minutes for local LLM processing...")
    payload = {
        "question": "Sebutkan hak asasi manusia menurut UUD 1945",
        "method": "native"
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout for local LLM
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Native ask test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting LawChain API Tests...")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    results = []
    
    # Test 1: Health check
    results.append(test_health())
    
    # Test 2: System info
    results.append(test_system_info())
    
    # Test 3: Ask with LangChain
    results.append(test_ask_endpoint_langchain())
    
    # Test 4: Ask with Native
    results.append(test_ask_endpoint_native())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    test_names = ["Health Check", "System Info", "LangChain Ask", "Native Ask"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
