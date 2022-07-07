from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tokenization
import data_collection
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import data_collection
import tokenization
import train_model


def domain_using_ml(tokenizer, use_input):
    model = keras.models.load_model('./models/domain.h5')
    sentence = [use_input]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    return model.predict(padded)


def intent_using_ml(tokenizer, use_input):
    model = keras.models.load_model('./models/intent.h5')
    sentence = [use_input]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    return model.predict(padded)


def tag_using_ml(tag_tokenizer_index, user_input):
    model = keras.models.load_model('./models/tag.h5')
    sentence = user_input
    words_for_test = tokenization.sentence_fit_on_text(tag_tokenizer_index, sentence)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)


def classify_using_ml_hotel(tag_tokenizer_index, user_input):
    model = keras.models.load_model('./models/classify_for_hotel.h5')
    words_for_test = tokenization.sentence_fit_on_text_in_list(tag_tokenizer_index, user_input)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)


def classify_using_ml_restaurant(tag_tokenizer_index, user_input):
    model = keras.models.load_model('./models/classify_for_restaurant.h5')
    words_for_test = tokenization.sentence_fit_on_text_in_list(tag_tokenizer_index, user_input)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)

# new use model code
def check_message_word_type_using_ml(user_input):
    model = keras.models.load_model('./models/transcript_model.h5')
    tokenizer = data_collection.collect_data_for_classification()
    words_for_test = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(words_for_test, maxlen=25, padding='post', truncating='post')
    predicted_word = []
    for index in range(25):
        if padded[0][index] != 0:
            predicted_word.append(model.predict({"transcript_input": padded, "word_input": np.stack([padded[0][index]])}))
        else:
            predicted_word.append(0)
    return predicted_word


def check_message_response_using_ml(user_input):
    model = keras.models.load_model('./models/response_or_not.h5')
    tokenizer = data_collection.collect_tokenizer_for_user()
    
    words_for_test = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(words_for_test, maxlen=25, padding='post', truncating='post')
    return model.predict(padded)


def check_message_stage_using_ml(user_input, response_or_not_for_stage):
    model = keras.models.load_model('./models/stage.h5')
    tokenizer = data_collection.collect_tokenizer_for_user()
    
    words_for_test = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(words_for_test, maxlen=25, padding='post', truncating='post')
    
    print(response_or_not_for_stage)
    return model.predict({"user_input": padded, "response_or_not_input": np.array([response_or_not_for_stage])})


def message_generate_using_ml(user_input, system_generated):
    model = keras.models.load_model('./models/system_response.h5')
    tokenizer_user = data_collection.collect_tokenizer_for_user()
    tokenizer_system = data_collection.collect_tokenizer_for_system()
    
    words_from_user = tokenizer_user.texts_to_sequences([user_input])
    words_for_system = tokenizer_system.texts_to_sequences([system_generated])
    
    padded_user_input = pad_sequences(words_from_user, maxlen=25, padding='post', truncating='post')
    padded_predicted_sentence = pad_sequences(words_for_system, maxlen=35, padding='post', truncating='post')
#     print(padded_user_input)
#     print(padded_predicted_sentence)
    all_prediction = model.predict({"user_input": padded_user_input, "system_accrue_message_input": padded_predicted_sentence})
    
    taget_index = 0
    for predict in all_prediction:
        for index in range(len(predict)):
            if predict[index] == max(predict):
                taget_index = index
   
    system_word = tokenizer_system.word_index
    system_word_list = list(system_word.keys())
    return system_word_list[taget_index-1]



