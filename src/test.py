import requests

url = 'http://localhost:9696/predict'

cv = {
  "department": "it",
  "job_title": "software_engineer",
  "location": "korea",
  "performance_rating": "rating2",
  "experience_years": 4,
  "status": "active",
  "work_mode": "on-site"
}

response = requests.post(url, json=cv)
predictions = response.json()

print(f'The candidates predicted annual salary is {format(predictions['annual_salary'], ".0f")} VND')