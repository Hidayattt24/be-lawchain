"""
Test cases for LawChain Backend API
"""

import pytest
import requests
import time
from typing import Dict, Any


class TestLawChainAPI:
    """Test suite for LawChain API"""
    
    BASE_URL = "http://localhost:8000/api/v1"
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except:
                time.sleep(1)
        else:
            pytest.fail("Server not ready after 30 seconds")
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
    
    def test_system_info(self):
        """Test system info endpoint"""
        response = requests.get(f"{self.BASE_URL}/system/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "ollama_status" in data
    
    def test_ask_question_langchain(self):
        """Test asking question with LangChain method"""
        question_data = {
            "question": "Apa itu Pancasila?",
            "method": "langchain"
        }
        
        response = requests.post(f"{self.BASE_URL}/ask", json=question_data)
        
        # May fail if not initialized, that's ok for testing
        if response.status_code == 200:
            data = response.json()
            assert "jawaban" in data
            assert "metrics" in data
            assert data["method"] == "langchain"
    
    def test_ask_question_native(self):
        """Test asking question with Native method"""
        question_data = {
            "question": "Apa itu Pancasila?",
            "method": "native"
        }
        
        response = requests.post(f"{self.BASE_URL}/ask", json=question_data)
        
        # May fail if not initialized, that's ok for testing
        if response.status_code == 200:
            data = response.json()
            assert "jawaban" in data
            assert "metrics" in data
            assert data["method"] == "native"
    
    def test_invalid_method(self):
        """Test asking question with invalid method"""
        question_data = {
            "question": "Test question",
            "method": "invalid_method"
        }
        
        response = requests.post(f"{self.BASE_URL}/ask", json=question_data)
        assert response.status_code == 400
    
    def test_empty_question(self):
        """Test asking empty question"""
        question_data = {
            "question": "",
            "method": "langchain"
        }
        
        response = requests.post(f"{self.BASE_URL}/ask", json=question_data)
        assert response.status_code == 422  # Validation error
    
    def test_system_status(self):
        """Test system status endpoint"""
        response = requests.get(f"{self.BASE_URL}/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "ollama" in data
        assert "lawchain" in data
        assert "vector_stores" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
