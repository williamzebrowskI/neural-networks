import requests
import json

def hit_api_and_parse_response(from_number, body, partner_id):
    url = "https://wyatt-openai-stage.bdtrust.org/sms"
    payload = {
        "From": from_number,
        "Body": body,
        "partnerId": partner_id
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Send the POST request to the API
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            api_response = response.json()
            
            messages_list = []
            
            # Extract the messages into the messages list
            for msg in api_response.get('messages', []):
                if 'message' in msg:
                    messages_list.append(msg['message'])
        
            
            # Prepare the response object
            response_object = {
                "messages": messages_list,
                "partnerId": api_response.get('partnerId', ''),
                "userId": api_response.get('user_id', '')
            }
            
            return response_object
        else:
            print(f"Failed to get a valid response, status code: {response.status_code}")
            return {"response": f"Error: Received status code {response.status_code}"}
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"response": f"Error parsing API response: {str(e)}"}

# Example usage
from_number = "22174962364"
body = "hi"
partner_id = "default"

result = hit_api_and_parse_response(from_number, body, partner_id)
print(result)