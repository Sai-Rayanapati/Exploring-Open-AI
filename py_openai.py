from openai import OpenAI
import pandas as pd
import requests
from pypdf import PdfReader

client = OpenAI(
    api_key="[Paste-your-key-here]")


def generate_email_reply(review):
    chat_transcript = [
        {
            "role": "system",
            "content": ("You are a helpful assistant for an e-commerce store. Generate polite, customer-friendly "
                        "emails that thank the customer for their purchase, address their concerns, and encourage "
                        "them to shop again. Do not give promotion codes or discounts to the customers. Do not "
                        "recommend other products. Keep the emails short. Do not give any contact information."
                        ),
        },
        {
            "role": "user",
            "content": (f"Here is a review from a customer:\n\n\"{review}\"\n\n"
                        "Please generate a polite and professional email to address their feedback."
                        ),
        },
    ]

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_transcript
    )

    return reply.choices[0].message.content


def generate_python_code(problem):
    chat_transcript = [
        {
            "role": "system",
            "content": ("You are a helpful assistant that generates Python code to solve programming problems. Ensure "
                        "the code is efficient, well-commented, and beginner-friendly."),
        },
        {
            "role": "user",
            "content": f"Write Python code to solve the following problem: {problem}",
        },
    ]

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_transcript
    )

    return reply.choices[0].message.content


def generate_text_summary(text):
    chat_transcript = [
        {
            "role": "system",
            "content": "You are an experienced Machine Learning research writer.",
        },
        {
            "role": "user",
            "content": f"Text to summarise: {text}. Summarise the above research paper in 1000 words:",
        },
    ]

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_transcript
    )

    return reply.choices[0].message.content


def generate_images(prompt):
    chat_transcript = [
        {
            "role": "system",
            "content": "You are a helpful assistant that improves image generation prompts for better results.",
        },
        {
            "role": "user",
            "content": f"Reword the following prompt for an image generation task: {prompt}",
        },
    ]
    prompt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_transcript
    )

    new_prompt = prompt_response.choices[0].message.content

    url_response = client.images.generate(
        model="dall-e-3",
        prompt=new_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = url_response.data[0].url

    image_response = requests.get(image_url)
    with open("generated_image.png", "wb") as file:
        file.write(image_response.content)
    print(f"Image saved as {"generated_image.png"}")


# Reviews
reviews = pd.read_csv('reviews.csv')

# Replies
columns = ['reviews', 'emails']
replies = pd.DataFrame(columns=columns)
replies['reviews'] = reviews.copy()

# Generate Email replies
replies['emails'] = replies['reviews'].apply(generate_email_reply)
replies.to_csv("email_replies.csv", index=False)

# Python problems
problems_df = pd.read_csv('python_problems.csv')
problems = problems_df['problems']

# Generate Python code
for problem in problems:
    generated_code = generate_python_code(problem)

    print(f"Problem: {problem.upper()}\n")
    print(f"Python code: \n{generated_code}\n")

# Summarise text
url = "https://arxiv.org/pdf/1706.03762.pdf"
ppr_data = requests.get(url).content

with open('paper.pdf', 'wb') as handler:
    handler.write(ppr_data)
reader = PdfReader("paper.pdf")
text = ""
for page in reader.pages[:2]:
    text += page.extract_text() + "\n"

paper_summary = generate_text_summary(text)

print(f"Paper summary:\n {paper_summary}")

# Generate images
prompt = "A serene mountain landscape with a crystal-clear lake, surrounded by pine forests under a golden sunrise."
generate_images(prompt)
