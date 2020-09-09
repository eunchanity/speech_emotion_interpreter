import streamlit as st
import pandas as pd
import numpy as np
import librosa
import json
import pickle
import keras
from keras.models import Sequential, Model, model_from_json
import speech_recognition as sr
from pydub import AudioSegment
import io
import pyaudio
import wave

# remove deprecation warning
st.set_option("deprecation.showfileUploaderEncoding", False)

# streamlit beta styling
st.beta_set_page_config(
    page_title="Speech Emotion Interpreter",
    page_icon=":ear:",
    layout="centered",
    initial_sidebar_state="expanded",
)
# streamlit outline
st.title("Speech Emotion Interpreter")
st.markdown("---")

# audio file upload
st.subheader("Upload Your Audio File Here (.wav):")
audio_file = st.file_uploader("", type=["wav"])
# audio player
st.audio(audio_file, format="audio/wav")

# load in model
json_file = open(
    "/Users/davidweon/davids_repo/projects/project5_speech_emotion/streamlit/saved_models/cnn_model_json.json",
    "r",
)
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(
    "/Users/davidweon/davids_repo/projects/project5_speech_emotion/streamlit/saved_models/cnn_model_weights.h5"
)

# optimizer
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)
st.write("")
st.write("")
if audio_file is not None:
    # transform new audio file data
    X, sample_rate = librosa.load(
        audio_file, res_type="kaiser_fast", duration=2.5, sr=44100, offset=0.5,
    )

    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    audio_df = pd.DataFrame(data=mfccs).T

    # model prediction
    audio_df = np.expand_dims(audio_df, axis=2)
    audio_df = loaded_model.predict(audio_df, batch_size=16, verbose=1)

    # map model prediction to saved label
    filename = "/Users/davidweon/davids_repo/projects/project5_speech_emotion/streamlit/saved_models/emotion_speech_labels"
    infile = open(filename, "rb")
    lb = pickle.load(infile)
    infile.close()

    # final predicted label
    final_label = audio_df.argmax(axis=1)
    final_label = final_label.astype(int).flatten()
    final_label = lb.inverse_transform((final_label))

    # extracting final label
    final_label = str(final_label)
    final_label = final_label.replace("[", "").replace("]", "").replace("'", "")
    split = final_label.split("_")
    st.write(f"The person that is speaking to you is **{split[0]}**.")
    st.write(f"The person that is speaking to you is emotionally **{split[1]}**.")

    # speech to text
    # r = sr.Recognizer()
    # audio = sr.AudioFile('path/wav_file.wav')
    # with audio as source:
    #     speech_recognition = r.record(source)

    # r.recognize_google(speech)

# inspirational quote
st.markdown("---")
st.markdown(
    "_Please use this as a supplementary tool. Be mindful of your surroundings and trust your intuition._ "
)
st.write("")
st.markdown(
    "_Every one of us is blind and deaf until our eyes are opened to our fellowmen, until our ears hear the voice of humanity.  \n-Helen Keller_"
)

# sidebar
st.sidebar.title("About This App")
st.sidebar.markdown(
    "This app processes input audio (.wav) and returns the speech emotion and speaker gender.  \n"
    "  \n"
    "_Speech transcription is in the app's code but could not be implemented due to Streamlit's limitations._  \n"
    "  \n"
    "The goal of this app is to create an alert system for hearing-impaired individuals that may need assistance in understanding speech intent and emotion."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Technical Details")
st.sidebar.markdown(
    "A 1D convoluted neural network (CNN) model was trained on the RAVDESS emotional speech dataset.  \n"
    "  \n"
    "The validation loss was `1.4470` and the validation accuracy was `0.525` for predicting gender _and_ emotion. The accuracy for gender and emotion individually differed quite a bit for this model. \n"
    "  \n"
    "_Read the in-depth analysis in my [blog post](https://eunchanity.github.io/speech-emotion/)._"
)
st.sidebar.subheader("Interpretation Capabilities")
st.sidebar.markdown(
    "  \n"
    "**Gender**: Male, Female  \n"
    "**Emotion**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise"
)

st.sidebar.subheader("References")
st.sidebar.markdown(
    "Dataset Used: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)"
)
st.sidebar.markdown("---")

st.sidebar.markdown("**Created by David Weon**")
st.sidebar.markdown(
    "[Link to App Github](https://github.com/eunchanity/speech_emotion_interpreter)  \n"
    "[Link to Blog Post](https://eunchanity.github.io/speech-emotion/)"
)
