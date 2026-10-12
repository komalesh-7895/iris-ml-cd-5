import unittest
from app import app


class TestIrisAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("prediction", response.get_json())

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_non_numeric_measurement(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": "abc",
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
