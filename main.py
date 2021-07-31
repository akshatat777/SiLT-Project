from text2sign import words_to_video
from audio_processing import load_mic
from audio_processing import text_to_speech
from signtotext import sign_to_text
from video2text_joint import videototext
import os, sys, subprocess

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def audiotosign():
    confirm = 'n'
    while confirm == 'n':
        text = load_mic()
        print("We think you said this: " + text)
        confirm = input("Confirm? (y/n) ")
    video_name = words_to_video(text)
    open_file(video_name)
    return video_name
#audiotosign()

def texttosign():
    text = input("What do you want in sign? ")
    video_name = words_to_video(text)
    open_file(video_name)
    return video_name

def signtotext():
    prediction = videototext()
    return prediction
#print(signtotext())

def signtoaudio():
    prediction = videototext()
    text_to_speech(prediction, "signs")
    open_file("signs.mp3")
signtoaudio()
