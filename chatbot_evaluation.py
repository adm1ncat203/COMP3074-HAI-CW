import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from chatbot import ChatBot


# Function to preprocess text (same as the one in chatbot.py)
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)


# Function to get an answer from the chatbot system
def get_answer(question):
    return chatbot.question_answering(question)


# Load config file
with open('./chatbot_config.json', 'r') as config_file:
    chatbot_config = json.load(config_file)

# Load the QA dataset
qa_dataset_path = "./dataset/COMP3074-CW1-Dataset-QA.csv"
qa_dataset = pd.read_csv(qa_dataset_path)

# Load the Transaction dataset (not used directly in this evaluation)
transaction_dataset_path = "./dataset/COMP3074-CW1-Dataset-Transaction.csv"
transaction_dataset = pd.read_csv(transaction_dataset_path)

# Initialize chatbot
chatbot = ChatBot(qa_dataset, transaction_dataset, chatbot_config)

# Define test set
test_set = [
    {"question": "what is the percentage of water in the human body?", "correct_answer": "57 percent in average"},
    {"question": "who is the author of white christmas?", "correct_answer": "Irving Berlin"},
    {"question": "who is the author of a rose is a rose is a rose?", "correct_answer": "Gertrude Stein"},
    {"question": "who wrote tree grows in brooklyn?", "correct_answer": "Betty Smith"},
    {"question": "Where is coldwater town?", "correct_answer": "Tate county"},
    {"question": "who invent radio?", "correct_answer": "Heinrich Rudolf Hertz"},
    {"question": "what is the number of people live in atlanta georgia?", "correct_answer": "432,427"},
    {"question": "when did big pokey born?", "correct_answer": "December 4, 1977"},
    {"question": "who is an actor in the movie leaving las vegas?", "correct_answer": "Nicolas Cage"},
    {"question": "what was the day that gary moore pass away?", "correct_answer": "6 February 2011"},
]

# Evaluate the QA system
correct_count = 0
for item in test_set:
    raw_system_answer = get_answer(item["question"])
    system_answer = preprocess_text(get_answer(item["question"]))
    correct_answer = preprocess_text(item["correct_answer"])

    print(f"Question: {item['question']}")
    # print(f"Expected: {correct_answer}")
    print(f"Expected (raw): {item['correct_answer']}")
    # print(f"Actual: {system_answer}")
    print(f"Actual (raw): {raw_system_answer}")
    print("-" * 10)

    # Compare answers (exact match or substring match)
    if correct_answer in system_answer or system_answer in correct_answer:
        correct_count += 1

# Calculate and print accuracy
accuracy = correct_count / len(test_set)
print(f"Accuracy: {accuracy:.2%}")
