from openai import OpenAI
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_images(image_path, meta_data):
    img = encode_image(image_path)
    img = f"data:image/jpeg;base64,{img}"
    img_quality = "low"

    message = meta_data["message"]
    premise = meta_data["premise"]
    strategy = meta_data["strategy"]

    system_prompt = "You are a helpful assistant designed to output JSON."
    turn_one_prompt = f"Attached is an image about \"{message}\". What message does this image intend to convey?"
    turn_two_prompt = f"The actual message that the image intended to convey is: \"{premise}\"."

    if strategy == "External Emotion":
        turn_two_prompt += "How well does your interpretation capture the feelings that this action may cause to other people as described in the intended message?"
    
    elif strategy == "Internal Emotion":
        turn_two_prompt += "How well does your interpretation capture the emotional reactions of the person who conducts this action as described in the intended message?"

    elif strategy == "Bandwagon":
        turn_two_prompt += "How well your interpretation capture the collective behavior of many people or poular opinion in the intended message?"

    elif strategy == "Consequence":
        turn_two_prompt += "How well does your interpretation capture the consequences of this action described in the intended message?"

    elif strategy == "Perceived Persona":
        turn_two_prompt += "How well does your interpretation capture the persona or attributes of the person who conducts this action perceived by other people as described in the intended message?"

    turn_two_prompt += f" Give a brief explanation in the \"reason\" key. Rate the score between 0 and 10 (0: not captured at all , 10: perfectly captured). "
    turn_two_prompt += "Provide your rating in the \"score\" key."

    # Initialize the conversation history
    conversation_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": turn_one_prompt
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": img,
                        "detail": img_quality
                    }
                },
            ],
        }
    ]

    # First turn (try it maximum 3 times)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_one = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                max_tokens=300,
            )
            break
        except:
            if attempt == (max_retries - 1):
                raise ValueError("Failed to get response from openai.")
            continue
    content_one = response_one.choices[0].message.content

    # Now turn two
    # Append the assistant's response from turn one to the conversation history
    conversation_history.append({
            "role": "assistant", 
            "content": [
                {
                    "type": "text",
                    "text": content_one
                }
            ]
        }
    )

    conversation_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": turn_two_prompt
                }
            ]
        }
    )
    #  Add system prompt to the front part of conversation history.
    #  It is needed for using the "response_format = json_object" parameter in the completion request.
    system_prompt_block = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt
            }
        ]
    }
    # insert the system prompt to the first index of the conversation history
    # The 0th index is not used to avoid giving image to the model
    conversation_history.insert(1, system_prompt_block)

    for attempt in range(max_retries):
        try:
            response_two = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=conversation_history[1:],
                max_tokens=300,
            )
        except:
            if attempt == (max_retries - 1):
                raise ValueError("Failed to get response from openai.")
            continue
    content_two = response_two.choices[0].message.content
    result = json.loads(content_two)
    score = result.get('score', None)
    reason = result.get('reason', None)
    
    if score is None:
        for attempt in range(max_retries):
            try:
                response_two = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=conversation_history[1:],
                    max_tokens=300,
                )
                content_two = response_two.choices[0].message.content
                result = json.loads(content_two)
                score = result.get('score', None)
                reason = result.get('reason', None)
                if score is not None:
                    score = int(score)
                    break
            except:
                if attempt == (max_retries - 1):
                    raise ValueError("Failed to get response from openai.")
                continue
    turn_one_prompt = str(turn_one_prompt)
    turn_two_prompt = str(turn_two_prompt)
    response_one = str(response_one)
    response_two = str(response_two)
    content_one = str(content_one)
    content_two = str(content_two)
    reason = str(reason)
    return score, reason, turn_one_prompt, turn_two_prompt, response_one, response_two, content_one, content_two

if __name__ == "__main__":
    try:
        # Please add openai api key to your env variable before trying this code
        # Or change the code to directly add the api key here
        client = OpenAI()
    except:
        raise ValueError("API key not loaded. Check the 'OPENAI_API_KEY' environment variable.")

    # Change the message, premise, strategy, and image path for your use case
    # We are using a sample image here
    message = "Do not smoke"
    premise = "Do not smoke. Then, you enjoy the freshness in your breath and clothes."
    strategy = "xReact"
    image_path = "./example_image.jpg"
    meta_data = {
        "message": message,
        "premise": premise,
        "strategy": strategy
    }
    score, reason, turn_one_prompt, turn_two_prompt, response_one, response_two, content_one, content_two = evaluate_images(image_path, meta_data)
    print(f"Score: {score}")
    print(f"Reason: {reason}")
