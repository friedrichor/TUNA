import os
import time
import requests


def call_gpt(model, messages, headers):
    data = {
        "model": model,
        "messages": messages,
        "n": 1,
        "max_tokens": 8192
    }

    answer = None
    while answer is None:
        try:
            r = requests.post(
                'https://api.openai.com/v1/chat/completions',
                json=data,
                headers=headers
            )
            resp = r.json()

            if r.status_code != 200 or 'choices' not in resp:
                print(f"Request failed: {resp}")
                continue

            message = resp['choices'][0]['message']
            answer = message['content']
            return (True, message, answer)
        except Exception as e:
            print(f"An exception occurred: {e}, retrying...")
            time.sleep(1)
