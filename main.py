# Import necessary libraries
import os
import speech_recognition as sr
from gtts import gTTS
import transformers
import time
import datetime
import numpy as np
import tensorflow as tf

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress the specific TensorFlow warning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Building the AI
class ChatBot:
    def __init__(self, name):
        self.text = None
        print("----- Starting up", name, "-----")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            self.text = "ERROR"
        try:
            self.text = recognizer.recognize_google(audio)
            print("Me  --> ", self.text)
        except sr.UnknownValueError:
            print("Me  -->  ERROR: Could not understand audio")
        except sr.RequestError as e:
            print(f"Me  -->  ERROR: Could not request results; {e}")

    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system('start res.mp3' if os.name == 'nt' else 'mpg123 res.mp3')
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        time.sleep(int(50 * duration))
        os.remove("res.mp3")

    def wake_up(self, text):
        return self.name.lower() in text.lower()

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


if __name__ == "__main__":
    ai = ChatBot(name="dev")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex = True
    while ex:
        ai.speech_to_text()

        if ai.wake_up(ai.text):
            res = "Hello I am Dave the AI, what can I do for you?"

        elif "time" in ai.text:
            res = ai.action_time()

        elif any(i in ai.text for i in ["thank", "thanks"]):
            res = np.random.choice(
                ["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "mention not"])
        elif any(i in ai.text for i in ["exit", "close"]):
            res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "peace out!"])
            ex = False

        else:
            if ai.text == "ERROR":
                res = "Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ") + 6:].strip()
        ai.text_to_speech(res)
    print("----- Closing down Dev -----")
