from openai import OpenAI
import requests
import re

def dalle_call(prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
            return response.data[0].url
        except:
            if attempt < max_retries - 1:
                continue
            else:
                raise ValueError("Failed to generate an image.")

def save_image(url):
    response = requests.get(url)
    safe_filename = re.sub(r'[^\w\-]', '', "generated_image")
    filename = f"./{safe_filename}.jpg"
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Image saved as {filename}")

def make_query(premise, message):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"I want to generate an image based on the theme: '{premise}' and the message: {message}. Please describe in detail how to represent this theme in an image. The result should be phrased as a complete sentence, emphasize the theme, and not exceed 10 sentences."
                }]
            )
            return response.choices[0].message.content
        except:
            if attempt < max_retries - 1:
                continue
            else:
                raise ValueError("Failed to make a query.")
            
def generate_image(premise, message):
    query = make_query(premise, message)
    prompt = f"Generate image: {query}. The generated image should be created in a photorealistic style."
    url = dalle_call(prompt)
    save_image(url)
    print(f"Image generated based on the theme: '{premise}' and the message: {message}")

if __name__ == "__main__":
    try:
        client = OpenAI()
    except:
        raise ValueError("API key not loaded. Check the 'OPENAI_API_KEY' environment variable.")
    
    # Example message and premise
    # Change these to your desired message and premise
    message = "Do not smoke"
    premise = "Do not smoke. Then, you enjoy the freshness in your breath and clothes."
    generate_image(premise, message)