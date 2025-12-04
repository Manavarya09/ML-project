import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.web_app import app

client = TestClient(app)

class TestWebApp(unittest.TestCase):
    @patch("src.web_app.predict_email_bert")
    def test_analyze_endpoint(self, mock_predict):
        # Mock return value matching the new format (label, proba, list of dicts)
        mock_predict.return_value = (1, 0.95, [
            {"word": "urgent", "score": 0.5, "start": 0, "end": 6},
            {"word": "verify", "score": 0.3, "start": 10, "end": 16}
        ])
        
        response = client.post("/analyze", data={"email_text": "urgent verify"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify structure
        self.assertTrue(data["is_phishing"])
        self.assertEqual(data["probability"], 0.95)
        self.assertEqual(len(data["explanation"]), 2)
        
        # Verify content
        self.assertEqual(data["explanation"][0]["word"], "urgent")
        self.assertEqual(data["explanation"][0]["start"], 0)
        self.assertEqual(data["explanation"][0]["end"], 6)
        self.assertEqual(data["explanation"][0]["score"], 0.5)

if __name__ == "__main__":
    unittest.main()
