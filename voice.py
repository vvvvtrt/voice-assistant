import pyttsx3


engine = pyttsx3.init()
engine.setProperty('rate', 180)


def speaker(text):
	'''Озвучка текста'''
	engine.say(text)
	engine.runAndWait()
