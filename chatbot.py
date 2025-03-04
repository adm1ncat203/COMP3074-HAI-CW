import json
import pandas as pd  # Dataset loading
import numpy as np
import re
import random
import pytz  # Timezone support
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer


# Required nltk resource
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class ChatBot:
    def __init__(self, qa_dataset, transaction_dataset, config):
        self.qa_data = qa_dataset
        self.user_name = None
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.qa_data['ProcessedQuestion'] = self.qa_data['Question'].apply(
            lambda x: self.preprocess_text(x)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.qa_data['ProcessedQuestion'])
        # self.tfidf_matrix = self.vectorizer.fit_transform(self.qa_data['Question'])
        self.keywords = config["keywords"]
        self.exit_keywords = config["exit_keywords"]
        self.similarity_threshold = config["similarity_threshold"]
        self.conversation_state = None  # Track context state
        self.selected_timezone = None  # Record the time zone selected by the user
        self.user_time_preference = None  # Stores user's preference for time (local or specific timezone)
        self.transaction_data = transaction_dataset
        self.transaction_data_grouped = self.transaction_data.groupby("TransactionType")
        self.transaction_state = 0
        self.transaction_details = {}

    def preprocess_text(self, text):
        """Preprocess text: lowercase, tokenize, remove stopwords, and lemmatize."""
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        return " ".join(tokens)

    def match_intent(self, user_input):
        """Match user input to the best intent."""
        # Light preprocessing for identity_management
        raw_input = user_input.lower().strip()
        # print(f"Debug: Raw input -> {raw_input}")

        # If currently in a transaction, continue handling the transaction
        if self.transaction_state != 0:  # Any non-zero state indicates an ongoing transaction
            if self.conversation_state == "plane_booking":
                return "transaction_plane_booking"
            elif self.conversation_state == "hotel_booking":
                return "transaction_hotel_booking"

        # Identity management
        if "my name is" in raw_input or "what is my name" in raw_input:
            return "identity_management"

        # Small talk
        if any(word in raw_input for word in self.keywords["small_talk_greetings"]):
            return "small_talk"
        if any(word in raw_input for word in self.keywords["small_talk_status"]):
            return "small_talk"
        if any(word in raw_input for word in self.keywords["small_talk_time"]):
            return "small_talk"
        if self.conversation_state in ["time_query", "timezone_query"]:
            return "small_talk"

        # Discoverability
        if any(word in raw_input for word in self.keywords["discoverability"]):
            return "discoverability"

        # Transactions x2
        if any(word in raw_input for word in self.keywords["transactions"]["plane_booking"]):
            self.conversation_state = "plane_booking"  # Set transaction type
            return "transaction_plane_booking"

        if any(word in raw_input for word in self.keywords["transactions"]["hotel_booking"]):
            self.conversation_state = "hotel_booking"  # Set transaction type
            return "transaction_hotel_booking"

        # Full preprocessing for intent matching
        processed_input = self.preprocess_text(user_input)

        if any(word in processed_input for word in self.exit_keywords):
            return "exit"

        for intent, words in self.keywords.items():
            if any(word in processed_input for word in words):
                return intent

        return "question_answering"

    def small_talk(self, user_input):
        """Handle small talk responses."""
        # Check time query
        raw_input = user_input.lower().strip()
        # print(f"Debug: Raw input -> {raw_input}")
        # Handle follow-up responses for time queries

        if self.conversation_state == "time_query":
            if "local" in raw_input:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.conversation_state = None  # Reset the conversation state
                self.user_time_preference = "local"  # Save user preference
                return f"The current local date and time is {current_time}."
            elif "timezone" in raw_input:
                self.conversation_state = "timezone_query"
                return "Which timezone would you like to know the time for? (e.g., 'Asia/Shanghai', 'UTC', 'US/Eastern')"
            else:
                return "I didn't understand that. Please specify 'local' or 'timezone'."

        # Handle timezone-specific query
        if self.conversation_state == "timezone_query":
            try:
                user_timezone = pytz.timezone(raw_input)
                current_time = datetime.now(user_timezone).strftime("%Y-%m-%d %H:%M:%S")
                self.conversation_state = None  # Reset the conversation state
                self.user_time_preference = raw_input  # Save user preference for the timezone
                return f"The current date and time in {raw_input.title()} is {current_time}."
            except pytz.UnknownTimeZoneError:
                return "I'm sorry, I couldn't recognize that timezone. Please try again."

        # Handle repeat queries if a time preference exists
        if any(word in raw_input for word in self.keywords["small_talk_time"]):
            if self.user_time_preference:
                if self.user_time_preference == "local":
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return f"The current local date and time is {current_time}."
                else:
                    try:
                        user_timezone = pytz.timezone(self.user_time_preference)
                        current_time = datetime.now(user_timezone).strftime("%Y-%m-%d %H:%M:%S")
                        return f"The current date and time in {self.user_time_preference.title()} is {current_time}."
                    except pytz.UnknownTimeZoneError:
                        self.user_time_preference = None  # Reset invalid preference
                        return "It seems your previously selected timezone is invalid. Please specify 'local' or 'timezone'."
            else:
                self.conversation_state = "time_query"
                return "Would you like to know the local time or the time in a specific timezone? Please reply 'local' or 'timezone'."

        # Check greetings
        if any(word in raw_input for word in self.keywords["small_talk_greetings"]):
            if self.user_name:
                return f"Hello again, {self.user_name}! How can I assist you today?"
            else:
                return "Hello! What's your name?"

        # Check status questions
        if any(word in raw_input for word in self.keywords["small_talk_status"]):
            return random.choice(self.keywords["small_talk_status_responses"])

        return "Hi there! Feel free to ask me anything."

    def identity_management(self, user_input):
        """Handle identity management."""
        raw_input = user_input.lower().strip()
        # print(f"Debug: Raw input -> {raw_input}")

        if not self.user_name and "my name is" in raw_input:
            # ([A-Za-z\s,\']+) is a capturing group that matches the actual name.
            # [A-Za-z]` allows letters in both uppercase and lowercase.
            # \s allows spaces (e.g., for multi-word names like "John Connor").
            # , allows commas (e.g., "Doe, John").
            # \' allows apostrophes (e.g., "O'Connor").
            # The + quantifier ensures that at least one character is matched in the name.
            # [.!?]? optionally matches a punctuation mark (exclamation mark, question mark) at the end of the input.
            self.user_name = re.search(r'my name is ([A-Za-z\s,\']+)[.!?]?', raw_input, re.IGNORECASE)
            if self.user_name:
                self.user_name = self.user_name.group(1).strip().title()  # Capitalize the first letter
                return f"Nice to meet you, {self.user_name}! How can I help you today?"
        if "what is my name" in raw_input:
            if self.user_name:
                return f"Your name is {self.user_name}. How can I help you today?"
            else:
                return "I don't know your name yet. Can you tell me your name?"
        return "I'm not sure how to handle that."

    def discoverability(self, user_input):
        """List chatbot features with or without personalization and prompt for name if not provided."""
        if self.user_name:
            # If the user's name is known, include it in the greeting
            greeting = f"I can assist you with the following tasks, {self.user_name}!\n"
        else:
            # If the user's name is not known, provide a general greeting and prompt for their name
            greeting = ("I can assist you with the following tasks. "
                        "But first, may I know your name?\n"
                        "You can tell me by saying, 'My name is [Your Name].'\n")

        return (
            f"{greeting}\n"
            "1. Answering questions using my knowledge base (e.g., 'What are stocks and bonds?').\n"
            "2. Learning and remembering your name (e.g., 'My name is Alice.').\n"
            "3. Providing the current date and time (e.g., 'What's the time?').\n"
            "4. Engaging in small talk (e.g., 'How are you doing?')\n"
            "5. Handling transactions like booking flights and hotels.\n\n"
            "Just let me know what you'd like to do!"
        )

    def question_answering(self, user_input):
        """Answer questions using the QA dataset."""
        # Full preprocessing for question answering
        user_input = self.preprocess_text(user_input)
        user_input_vectorized = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_input_vectorized, self.tfidf_matrix).flatten()
        # print(f"Debug: Processed input for QA -> {user_input}")
        # print(f"Debug: Similarities -> {similarities}")

        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]

        if best_match_score > self.similarity_threshold:
            # matched_question = self.qa_data.iloc[best_match_idx]['ProcessedQuestion']
            answer = self.qa_data.iloc[best_match_idx]['Answer']
            # return f"Cosine Similiarities: {similarities[best_match_idx]} \nAnswer: {answer}"
            return f"{answer}"

        # If similarity is too low, offer suggestions or feedback
        sorted_indices = similarities.argsort()[::-1]  # Sort by descending similarity
        # Adjust threshold (0.5 / 2) for suggestions
        suggested_questions = [
            self.qa_data.iloc[idx]['Question']
            for idx in sorted_indices if similarities[idx] > (self.similarity_threshold / 2)
        ]

        if suggested_questions:
            return (
                    "I'm sorry, I don't have an exact answer. "
                    f"Did you mean one of these questions?\n"
                    + "\n".join(f"- {q}" for q in suggested_questions[:3])  # Limit suggestions
            )
        else:
            return (
                "I'm sorry, I couldn't find any related questions in my knowledge base. "
                "Please try rephrasing your question or ask about a different topic."
            )

    def handle_plane_booking(self, user_input):
        """Handle the plane booking transaction with validations."""
        # Filter plane_booking questions
        plane_data = self.transaction_data_grouped.get_group("plane_booking")

        # Initial confirmation step
        if self.transaction_state == 0:
            self.conversation_state = "plane_booking"
            self.transaction_state = -1  # Temporary state for confirmation
            return (
                "You would like to book a plane ticket. Is that correct?\n"
                "Please reply 'yes'/'y' to proceed or 'no'/'n' to cancel."
            )

        # Confirmation step
        if self.transaction_state == -1:
            if user_input.lower() in ["yes", "y"]:
                self.transaction_state = 1
                return plane_data.iloc[self.transaction_state - 1]["Question"]
            elif user_input.lower() in ["no", "n"]:
                self.reset_transaction_state()
                return "Alright, let me know if there's anything else I can help you with."
            else:
                return "I didn't understand that. Please reply with 'yes'/'y' to proceed or 'no'/'n' to cancel."

        # Handle step-by-step booking
        if 1 <= self.transaction_state <= len(plane_data):
            question_key = f"step_{self.transaction_state}"
            # Preprocess user input
            processed_input = self.preprocess_text(user_input)
            # Validate user input
            if self.transaction_state == 1 or self.transaction_state == 2:  # Locations
                if not processed_input.replace(" ", "").isalpha():
                    return "That doesn't look like a valid location. Please enter a valid city or airport name."
            elif self.transaction_state == 3 or self.transaction_state == 4:  # Dates
                if not self.validate_date_format(user_input):
                    return "That doesn't look like a valid date. Please enter the date in YYYY-MM-DD format."
            elif self.transaction_state == 5:  # Number of passengers
                if not user_input.isdigit() or int(user_input) <= 0:
                    return "Please enter a valid number of passengers (e.g., 1, 2, 3)."
            elif self.transaction_state == 6:  # Class type
                valid_classes = ["economy", "premium economy", "business", "first class"]
                if processed_input.lower() not in valid_classes:
                    return f"Please choose a valid option: {', '.join(valid_classes)}."
            # Save valid input
            self.transaction_details[question_key] = user_input.strip()

            # Move to the next step
            if self.transaction_state < len(plane_data):
                self.transaction_state += 1
                return plane_data.iloc[self.transaction_state - 1]["Question"]
            else:
                # Final confirmation step
                self.transaction_state = -2
                return (
                    f"Thank you! Here is your flight booking detail:\n"
                    f"- From: {self.transaction_details.get('step_1', 'N/A')}\n"
                    f"- To: {self.transaction_details.get('step_2', 'N/A')}\n"
                    f"- Departure: {self.transaction_details.get('step_3', 'N/A')}\n"
                    f"- Return: {self.transaction_details.get('step_4', 'N/A')}\n"
                    f"- Passengers: {self.transaction_details.get('step_5', 'N/A')}\n"
                    f"- Class: {self.transaction_details.get('step_6', 'N/A')}\n"
                    "Would you like to confirm this booking? (yes/y or no/n)"
                )

        # Final confirmation
        if self.transaction_state == -2:
            if user_input.lower() in ["yes", "y"]:
                self.reset_transaction_state()
                return "Your flight booking has been confirmed! Thank you for using our service."
            elif user_input.lower() in ["no", "n"]:
                self.reset_transaction_state()
                return "Your booking has been canceled. Let me know if there's anything else I can assist you with."
            else:
                return "I didn't understand that. Please reply with 'yes'/'y' to confirm or 'no'/'n' to cancel."

        # Fallback
        return "I'm sorry, something went wrong. Can you try again?"

    def handle_hotel_booking(self, user_input):
        """Handle the hotel booking transaction with validations."""
        # Filter hotel_booking questions
        hotel_data = self.transaction_data_grouped.get_group("hotel_booking")

        if self.transaction_state == 0:
            self.conversation_state = "hotel_booking"
            self.transaction_state = -1
            return (
                "You would like to book a hotel. Is that correct?\n"
                "Please reply 'yes'/'y' to proceed or 'no'/'n' to cancel."
            )

        if self.transaction_state == -1:
            if user_input.lower() in ["yes", "y"]:
                self.transaction_state = 1
                return hotel_data.iloc[self.transaction_state - 1]["Question"]
            elif user_input.lower() in ["no", "n"]:
                self.reset_transaction_state()
                return "Alright, let me know if there's anything else I can help you with."
            else:
                return "I didn't understand that. Please reply with 'yes'/'y' to proceed or 'no'/'n' to cancel."

        if 1 <= self.transaction_state <= len(hotel_data):
            question_key = f"step_{self.transaction_state}"
            # Preprocess user input
            processed_input = self.preprocess_text(user_input)
            # Validate user input
            if self.transaction_state == 1:  # Location
                if not processed_input.replace(" ", "").isalpha():
                    return "That doesn't look like a valid location. Please enter a valid city name."
            elif self.transaction_state in [2, 3]:  # Dates
                if not self.validate_date_format(user_input):
                    return "That doesn't look like a valid date. Please enter the date in YYYY-MM-DD format."
            elif self.transaction_state == 4:  # Number of guests
                if not user_input.isdigit() or int(user_input) <= 0:
                    return "Please enter a valid number of guests (e.g., 1, 2, 3)."
            elif self.transaction_state == 5:  # Room type
                valid_rooms = ["single", "double", "suite"]
                if processed_input.lower() not in valid_rooms:
                    return f"Please choose a valid option: {', '.join(valid_rooms)}."
            # Save valid input
            self.transaction_details[question_key] = user_input.strip()

            if self.transaction_state < len(hotel_data):
                self.transaction_state += 1
                return hotel_data.iloc[self.transaction_state - 1]["Question"]
            else:
                self.transaction_state = -2
                return (
                    f"Thank you! Here is your hotel booking detail:\n"
                    f"- Location: {self.transaction_details.get('step_1', 'N/A')}\n"
                    f"- Check-in: {self.transaction_details.get('step_2', 'N/A')}\n"
                    f"- Check-out: {self.transaction_details.get('step_3', 'N/A')}\n"
                    f"- Guests: {self.transaction_details.get('step_4', 'N/A')}\n"
                    f"- Room type: {self.transaction_details.get('step_5', 'N/A')}\n"
                    "Would you like to confirm this booking? (yes/y or no/n)"
                )

        if self.transaction_state == -2:
            if user_input.lower() in ["yes", "y"]:
                self.reset_transaction_state()
                return "Your hotel booking has been confirmed! Thank you for using our service."
            elif user_input.lower() in ["no", "n"]:
                self.reset_transaction_state()
                return "Your booking has been canceled. Let me know if there's anything else I can assist you with."
            else:
                return "I didn't understand that. Please reply with 'yes'/'y' to confirm or 'no'/'n' to cancel."

        return "I'm sorry, something went wrong. Can you try again."

    def validate_date_format(self, date_str):
        """Validate if the input date is in the correct YYYY-MM-DD format."""
        try:
            date_str = date_str.strip()
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def reset_transaction_state(self):
        """Reset transaction-related states."""
        self.transaction_state = 0
        self.conversation_state = None
        self.transaction_details = {}

    def get_response(self, user_input):
        """Route user input to the appropriate function."""
        # Check if the input is empty or contains only spaces
        if not user_input.strip():
            return "Your input seems to be empty. Please enter valid information."
        # Determine the user's intent
        intent = self.match_intent(user_input)
        if intent == "exit":
            # Reset transaction state if user exits during a transaction
            self.transaction_state = 0
            self.transaction_details = {}
            return "exit"  # Special flag for exiting
        # small talk
        elif intent == "small_talk":
            return self.small_talk(user_input)
        # discoverability
        elif intent == "discoverability":
            return self.discoverability(user_input)
        # identity management
        elif intent == "identity_management":
            return self.identity_management(user_input)
        # transactions (plane booking & hotel booking)
        elif intent == "transaction_plane_booking":
            return self.handle_plane_booking(user_input)
        elif intent == "transaction_hotel_booking":
            return self.handle_hotel_booking(user_input)
        # question answering
        elif intent == "question_answering":
            return self.question_answering(user_input)
        else:
            return "I'm not sure how to respond to that. Can you rephrase or try another question?"


if __name__ == "__main__":
    # Load datasets
    qa_dataset = pd.read_csv("./dataset/COMP3074-CW1-Dataset-QA.csv")
    transaction_dataset = pd.read_csv("./dataset/COMP3074-CW1-Dataset-Transaction.csv")
    # Load configuration file for keywords and other parameters
    with open("./chatbot_config.json", 'r') as config_file:
        config = json.load(config_file)

    # Instantiate the chatbot
    chatbot = ChatBot(qa_dataset, transaction_dataset, config)

    # Command-line loop for interaction
    print("\nChatBot is now running. Type 'exit', 'goodbye', or 'bye' to quit.")
    while True:
        user_input = input(">>>You: ").strip()
        response = chatbot.get_response(user_input)
        if response == "exit":  # Special flag for exiting
            print(">>>ChatBot: Goodbye! Have a great day!\n")
            break
        print(f">>>ChatBot: {response}")
