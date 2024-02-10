from sklearn.feature_extraction.text import CountVectorizer  # pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import sounddevice as sd  # pip install sounddevice
import vosk  # pip install vosk
import torch

import json
import queue

import words
from skills import *
import speech_generation
import voice

q = queue.Queue()

model = vosk.Model('model_small')


device = sd.default.device
samplerate = int(sd.query_devices(device[0], 'input')['default_samplerate'])

dialogue_ = False
gen_ = False
count_req = 0

def callback(indata, frames, time, status):
    q.put(bytes(indata))


def recognize(data, vectorizer, clf):
    global dialogue_, gen_, count_req

    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        return

    if count_req >= 5:
        chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
        count_req = 0

    data.replace(list(trg)[0], '')
    print(data[4:], dialogue_)

    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]

    count_req += 1

    func_name = answer.split()[0]

    if dialogue_ and func_name != "stop":
        voice.speaker(speech_generation.answers(data[4:], "2"))
        return
    elif gen_:
        voice.speaker(speech_generation.answers(data[4:], "3"))
        return


    if answer.split()[0] == "short":
        # a = answer.replace(func_name, '')
        voice.speaker(speech_generation.answers(data[4:], "1"))
    elif answer.split()[0] == "dialogue":
        dialogue_ = True
        chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
        voice.speaker(answer.replace(func_name, ''))
    elif answer.split()[0] == "stop":
        dialogue_ = False
        voice.speaker(answer.replace(func_name, ''))
    elif answer.split()[0] == "gen":
        gen_ = True
        voice.speaker(answer.replace(func_name, ''))
    else:
        print(answer.split()[0])
        voice.speaker(answer.replace(func_name, ''))
        exec(func_name + '()')
        print(dialogue_)


def main():
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))

    del words.data_set

    with sd.RawInputStream(samplerate=samplerate, blocksize=16000, device=device[0], dtype='int16',
                           channels=1, callback=callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())['text']
                recognize(data, vectorizer, clf)
            # else:
            #     print(rec.PartialResult())


if __name__ == '__main__':
    main()
