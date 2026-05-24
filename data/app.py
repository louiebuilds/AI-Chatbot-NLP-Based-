import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("data/faq.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

# Chat loop
print("Chatbot is running! Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print("Bot:", response)
