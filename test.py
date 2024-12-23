from fastapi.testclient import TestClient
from app import app  # Replace 'main' with the name of your FastAPI app module

client = TestClient(app)

# Sample input data with multiple data points
sample_data = {
    "data": [
        {
            "age": 35,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "duration": 180,
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp_var_rate": -1.8,
            "cons_price_idx": 93.994,
            "cons_conf_idx": -36.4,
            "euribor3m": 4.857,
            "nr_employed": 5099.1,
        },
        {
            "age": 42,
            "job": "technician",
            "marital": "single",
            "education": "high.school",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "telephone",
            "month": "jul",
            "day_of_week": "fri",
            "duration": 210,
            "campaign": 2,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp_var_rate": -2.0,
            "cons_price_idx": 92.893,
            "cons_conf_idx": -42.7,
            "euribor3m": 4.191,
            "nr_employed": 5076.2,
        }
    ]
}

def test_endpoint_running_and_handling_multiple_data():
    """
    Test if the /predict endpoint is running and can handle multiple data points.
    """
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200  # Ensure the endpoint is running
    json_response = response.json()
    assert "predictions" in json_response  # Ensure the response contains predictions
    assert isinstance(json_response["predictions"], list)  # Predictions should be a list
    assert len(json_response["predictions"]) == len(sample_data["data"])  # Should match input count
