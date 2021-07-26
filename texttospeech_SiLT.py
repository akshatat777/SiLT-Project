from gtts import gTTS
import os

def text_to_speech(input_text, save_name: str):
    """Enter the text to convert to speech
       Enter the file name you want to save."""
    language = "en"
    myobject = gTTS(text=input_text, lang=language, slow=False)
    myobject.save(str(save_name)+str(".mp3"))
    os.system(str(save_name)+str(".mp3"))

text_to_speech("Hi how are you doing today? I am a text to speech assistant.","helloworld")