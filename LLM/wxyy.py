import requests
import json

# Replace these values with your actual credentials
app_id = "xxx"  # Your appId
secret_key = "xxx"  # Your secretKey
source = "xxx"  # Your source identifier
open_id = "xxx"  # Your openId

# The endpoint URL
url = "https://agentapi.baidu.com/assistant/conversation"

# Headers for the request
headers = {
    "Content-Type": "application/json",
}

# Payload (data to send in the request)
payload = {
    "message": {"content": {"type": "text", "value": {"showText": "你好"}}},
    "source": source,
    "from": "openapi",
    "openId": open_id,
}

# Make the POST request to the API
response = requests.post(
    url,
    params={"appId": app_id, "secretKey": secret_key},
    headers=headers,
    data=json.dumps(payload),
)

# Check if the request was successful
if response.status_code == 200:
    print("Response from Wenxin Yiyan API:", response.json())
else:
    print("Error:", response.status_code, response.text)
