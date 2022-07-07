import json
from bs4 import BeautifulSoup
import tokenization
import random
import re


# collect data from json
def raw_domain_collect():   # 1, Domain identification, select database
    hotel_message, restaurant_message, domain_label = [], [], []
    domain_from_j = open('./dataset_for_domain.json', 'r')
    domain_data_dic = json.loads(domain_from_j.read())
    for data in domain_data_dic:
        for turn in data['dialogue']:
            if turn['domain'] == "hotel":
                hotel_message.append(turn['transcript'])
            elif turn['domain'] == "restaurant":
                restaurant_message.append(turn['transcript'])
    for i in range(len(hotel_message) + len(restaurant_message)):
        if i < len(hotel_message):
            domain_label.append(0) # label 0 = hotel
        else:
            domain_label.append(1)  # label 1 = restaurant
    domain_from_j.close()
    return (hotel_message + restaurant_message), domain_label


def raw_intent_collect():   # 2, Intent Detection   ->  check which tags (search by what?)
    hotel_intent, restaurant_intent, total_message = [], [], []
    label_list, hotel_message, restaurant_message, message_label_in_one_hot = [], [], [], []
    print("2. Collecting data and set it as list")
    domain_from_j = open('./dataset_for_domain.json', 'r')
    domain_data_dic = json.loads(domain_from_j.read())
    for data in domain_data_dic:    # collect message and turn_label from dialogue
        for turn in data['dialogue']:
            if turn['domain'] == "hotel":
                hotel_message.append(turn['transcript'])
                hotel_intent.append(turn['turn_label'])
            elif turn['domain'] == "restaurant":
                restaurant_message.append(turn['transcript'])
                restaurant_intent.append(turn['turn_label'])
    total_message = hotel_message + restaurant_message
    total_intent = hotel_intent + restaurant_intent

    for intent_list in total_intent:    # create one-hot
        for pair in intent_list:
            if pair[0] not in label_list:
                label_list.append(pair[0])

    # one-hot label (3 degrees)
    for intent_list in total_intent:    # create one-hot for message_label_in_one_hot list
        one_hot = []
        for _ in range(4):
            one_hot.append(0)

        for pair in intent_list:
            if pair[0] == 'hotel-pricerange' or pair[0] == 'restaurant-pricerange':     # search by price
                one_hot[0] = 1
            elif pair[0] == 'hotel-area' or pair[0] == 'restaurant-area':               # search by area
                one_hot[1] = 1
            elif pair[0] == 'hotel-name' or pair[0] == 'restaurant-name':               # search by name
                one_hot[2] = 1
            elif pair[0] == 'restaurant-food':                                          # search by food type
                one_hot[3] = 1
        message_label_in_one_hot.append(one_hot)
    return total_message, message_label_in_one_hot  # return the message and the one-hot label


def raw_collect_tag_data():  # 3, Slot Filling  -> # use in RNN for key_word detection
    print("3, Collect word need to be attend")
    att_word_in_message, no_care_in_message = [], []    # for tag detection
    care_word, no_care_word = [], []
    dic_key_list = []   # tag from message in json

    message_from_j_with_tag = open('./MultiWOZ_1/data.json', 'r')
    message_with_tag_dic = json.loads(message_from_j_with_tag.read())
    for i in range(len(message_with_tag_dic)):  # read all data from .json
        key_from_dic = list(message_with_tag_dic.keys())[i]
        dic_key_list.append(str(key_from_dic))

    for key_json in dic_key_list:
        for message in message_with_tag_dic[key_json]['goal']['message']:
            soup = BeautifulSoup(str(message), "html.parser")
            item_text = soup.find_all()
            if len(item_text) != 0:
                soup2 = BeautifulSoup(str(item_text), "html.parser")
                item_text = soup2.find('span', attrs={'class': 'emphasis'})
                att_word_in_message.append(item_text.text)
    att_word_in_message = remove_repeat(att_word_in_message)

    for key_json in dic_key_list:    # select message without tags
        for message in message_with_tag_dic[key_json]['goal']['message']:
            no_care_message_in_turn = ""
            for word in message:
                if word == "<":
                    no_care_in_message.append(no_care_message_in_turn)
                    break
                else:
                    no_care_message_in_turn += word

    for string in att_word_in_message:
        for word in string.split():
            care_word.append(word)
    res = [i for n, i in enumerate(care_word) if i not in care_word[:n]]
    care_word = res

    for string in no_care_in_message:
        for word in string.split():
            no_care_word.append(word)
    message_from_j_with_tag.close()

    attention_word_label = []
    for i in range(len(care_word)+len(no_care_word)):
        if i < len(care_word):
            attention_word_label.append(1)
        else:
            attention_word_label.append(0)
    return (care_word + no_care_word), attention_word_label


def drain_message():
    print("3.tag by new message collect")
    # read
    my_json = open('./MultiWOZ_1/data.json', 'r')
    json_data = my_json.read()

    # parse
    obj = json.loads(json_data)
    watch_list = []
    dun_care_list = []

    # Place into lists
    for x in obj:
        for y in obj[x]['goal']['message']:
            y = str(y)

            # get strings between the tags
            opening_tag = "span class='emphasis'"
            closing_tag = "span"
            reg_str = "<" + opening_tag + ">(.*?)</" + closing_tag + ">"
            result = re.findall(reg_str, y)
            reduced = y
            for z in result:
                remove_str = "<span class='emphasis'>" + z + "</span>"
                watch_list.append(z)
                reduced = reduced.replace(remove_str, "")

            dun_care_list.append(reduced)

    # remove duplicate items in string
    watch_list = list(dict.fromkeys(watch_list))
    dun_care_list = list(dict.fromkeys(dun_care_list))

    # for word only
    care_word_list, not_care_word_list = [], []
    for sentence in dun_care_list:
        word_in_sentence = sentence.split()
        for word in word_in_sentence:
            if word not in not_care_word_list:
                not_care_word_list.append(word)
    for sentence in watch_list:
        word_in_sentence = sentence.split()
        for word in word_in_sentence:
            if word not in care_word_list and word not in not_care_word_list:
                care_word_list.append(word)

    tag_words = care_word_list + not_care_word_list
    tag_labels = []
    for _ in range(len(care_word_list)):
        tag_labels.append(1)
    for _ in range(len(not_care_word_list)):
        tag_labels.append(0)
    return tag_words, tag_labels


def drain_no_care_word():
    print("4.1.tag by new message collect")
    # read
    my_json = open('./MultiWOZ_1/data.json', 'r')
    json_data = my_json.read()

    # parse
    obj = json.loads(json_data)
    watch_list = []
    dun_care_list = []

    # Place into lists
    for x in obj:
        for y in obj[x]['goal']['message']:
            y = str(y)

            # get strings between the tags
            opening_tag = "span class='emphasis'"
            closing_tag = "span"
            reg_str = "<" + opening_tag + ">(.*?)</" + closing_tag + ">"
            result = re.findall(reg_str, y)
            reduced = y
            for z in result:
                remove_str = "<span class='emphasis'>" + z + "</span>"
                watch_list.append(z)
                reduced = reduced.replace(remove_str, "")

            dun_care_list.append(reduced)

    # remove duplicate items in string
    watch_list = list(dict.fromkeys(watch_list))
    dun_care_list = list(dict.fromkeys(dun_care_list))

    # for word only
    not_care_word_list = []
    for sentence in dun_care_list:
        word_in_sentence = sentence.split()
        for word in word_in_sentence:
            if word not in not_care_word_list and word not in watch_list:
                not_care_word_list.append(word)
    return not_care_word_list


def raw_key_word_collect_for_hotel():
    message_with_keyword = []
    m_price, m_name, m_location = [], [], []

    print("4.2. key_word_collect for hotel")
    domain_from_j = open('./dataset_for_domain.json', 'r')
    domain_data_dic = json.loads(domain_from_j.read())
    for data in domain_data_dic:    # collect message and turn_label from dialogue
        for turn in data['dialogue']:
            if turn['turn_label']:
                message_with_keyword.append(turn['turn_label'])

    for message_in_turn in message_with_keyword:    # set output_dim
        for label_transcript in message_in_turn:
            if label_transcript[0] == 'hotel-pricerange' or label_transcript[0] == 'restaurant-pricerange':
                m_price.append(label_transcript[1])
            elif label_transcript[0] == 'train-destination' or label_transcript[0] == 'train-departure' \
                    or label_transcript[0] == 'taxi-destination' or label_transcript[0] == 'taxi-departure':
                m_location.append(label_transcript[1])
            elif label_transcript[0] == 'hotel-name' or label_transcript[0] == 'attraction-name':
                m_name.append(label_transcript[1])

    keyword_message, keyword_label = [], []     # return for train
    keyword_message = m_price + m_location + m_name

    output_dim = 3
    for word in keyword_message:
        one_hot = []
        for _ in range(output_dim):
            one_hot.append(0)

        if word in m_price:
            one_hot[0] = 1
        elif word in m_location:
            one_hot[1] = 1
        elif word in m_name:
            one_hot[2] = 1
        keyword_label.append(one_hot)
    return keyword_message, keyword_label


def raw_key_word_collect_for_restaurant():
    message_with_keyword = []
    m_price, m_location, m_name, m_food = [], [], [], []

    domain_from_j = open('./dataset_for_domain.json', 'r')
    domain_data_dic = json.loads(domain_from_j.read())
    for data in domain_data_dic:    # collect message and turn_label from dialogue
        for turn in data['dialogue']:
            if turn['turn_label']:
                message_with_keyword.append(turn['turn_label'])

    for message_in_turn in message_with_keyword:
        for label_transcript in message_in_turn:
            if label_transcript[0] == 'hotel-pricerange' or label_transcript[0] == 'restaurant-pricerange':
                m_price.append(label_transcript[1])
            elif label_transcript[0] == 'train-destination' or label_transcript[0] == 'train-departure' \
                    or label_transcript[0] == 'taxi-destination' or label_transcript[0] == 'taxi-departure':
                m_location.append(label_transcript[1])
            elif label_transcript[0] == 'hotel-name' or label_transcript[0] == 'attraction-name':
                m_name.append(label_transcript[1])
            elif label_transcript[0] == 'restaurant-food':
                m_food.append(label_transcript[1])

    keyword_message, keyword_label = [], []     # return
    keyword_message = m_price + m_location + m_name + m_food
    for word in keyword_message:
        one_hot = []
        for _ in range(4):
            one_hot.append(0)

        if word in m_price:
            one_hot[0] = 1
        elif word in m_location:
            one_hot[1] = 1
        elif word in m_name:
            one_hot[2] = 1
        elif word in m_food:
            one_hot[3] = 1
        keyword_label.append(one_hot)
    return keyword_message, keyword_label


def remove_repeat(list_collected):
    res = [i for n, i in enumerate(list_collected) if i not in list_collected[:n]]
    list_collected = res
    return list_collected


# collect the data for ML train
def collect_domain_model_data():
    # domain model
    domain_message, domain_label = raw_domain_collect()  # collect data ->  domain_message = intent_message
    domain_message_train_in_str, domain_message_test_in_str = domain_message[:28000], domain_message[28000:]

    model_data = tokenization.tokenize_message(domain_message_train_in_str, domain_message_test_in_str)
    domain_tokenizer, domain_message_train, domain_message_test = model_data[0], model_data[1], model_data[2]

    domain_model_train, domain_model_test = [domain_message_train, domain_label[:28000]], [domain_message_test, domain_label[28000:]]  # train
    return domain_tokenizer, domain_model_train, domain_model_test


def collect_intent_data():
    # intent model
    intent_message, intent_label = raw_intent_collect()     # collect data
    intent_message_train_in_str = intent_message[:28000]
    intent_message_test_in_str = intent_message[28000:]

    model_data = tokenization.tokenize_message(intent_message_train_in_str, intent_message_test_in_str)  # tokenization
    intent_tokenizer, intent_message_train, intent_message_test = model_data[0], model_data[1], model_data[2]

    # train
    intent_model_train, intent_model_test = [intent_message_train, intent_label[:28000]], [intent_message_test, intent_label[28000:]]
    return intent_tokenizer, intent_model_train, intent_model_test


def collect_tag_data():
    tag_words, tag_label = drain_message()     # collect data
    print(len(tag_words), len(tag_label))
    train_words_in_string, test_words_list_in_string, train_label, test_label = random_collect_test_data(tag_words, tag_label, 80)

    tag_words_in_int = tokenization.tokenize_char_level(train_words_in_string, test_words_list_in_string)
    tag_tokenizer_index, tag_words_train, tag_words_test = tag_words_in_int[0], tag_words_in_int[1], tag_words_in_int[2]

    # train
    tag_model_train, tag_model_test = [tag_words_train, train_label], [tag_words_test, test_label]
    return tag_tokenizer_index, tag_model_train, tag_model_test


def collect_classify_data_for_hotel():
    tag_words, tag_label = raw_key_word_collect_for_hotel()    # collect data
    # collect random data
    train_words_in_string, test_words_list_in_string, train_label, test_label = random_collect_test_data(tag_words,tag_label,80)

    # tokenization
    classify_words_in_int = tokenization.tokenize_char_level(train_words_in_string, test_words_list_in_string) # tokenization
    classify_tokenizer, classify_words_train, classify_words_test = classify_words_in_int[0], classify_words_in_int[1], classify_words_in_int[2]

    classify_model_train, classify_model_test = [classify_words_train, train_label], [classify_words_test, test_label]
    return classify_tokenizer, classify_model_train, classify_model_test


def collect_classify_data_for_restaurant():
    tag_words, tag_label = raw_key_word_collect_for_restaurant()    # collect data
    # collect random data
    train_words_in_string, test_words_list_in_string, train_label, test_label = random_collect_test_data(tag_words,tag_label,80)

    # tokenization
    classify_words_in_int = tokenization.tokenize_char_level(train_words_in_string, test_words_list_in_string) # tokenization
    classify_tokenizer, classify_words_train, classify_words_test = classify_words_in_int[0], classify_words_in_int[1], classify_words_in_int[2]

    classify_model_train, classify_model_test = [classify_words_train, train_label], [classify_words_test, test_label]
    return classify_tokenizer, classify_model_train, classify_model_test


def random_collect_test_data(data_string, label, test_data_size):
    # collect random test
    train_words_in_string, train_label = [], []
    test_words_list_in_string, test_label = [], []
    collected_index_list = []

    while len(test_words_list_in_string) != int(test_data_size):  # collect 100 test data
        rand_idx = random.randrange(len(data_string))
        random_data = data_string[rand_idx]
        random_label = label[rand_idx]
        if random_data not in test_words_list_in_string:    # prevent repeat
            test_words_list_in_string.append(random_data)
            test_label.append(random_label)
            collected_index_list.append(rand_idx)

    for index in range(len(data_string)):
        if index not in collected_index_list:
            train_words_in_string.append(data_string[index])
            train_label.append(label[index])
    return train_words_in_string, test_words_list_in_string, train_label, test_label


# dataset for text generation
def collect_conversation():
    data_from_j = open('./dataset_for_domain.json', 'r')
    json_data_dic = json.loads(data_from_j.read())

    # for hotel data collection
    hotel_conversation_transcript = []
    for data in json_data_dic:
        if len(data['domains']) != 1:   # clean data not in the same region
            continue
        turn_conversation_transcript = []
        for turn in data['dialogue']:
            if turn['domain'] == "hotel":
                turn_id = turn['turn_idx']
                hotel_conversation_turn_transcript = turn['transcript']
                hotel_conversation_turn_system_transcript = turn['system_transcript']
                turn_conversation_transcript.append({'hotel_turn_id': str(turn_id),
                                                     'hotel_user_transcript': hotel_conversation_turn_transcript,
                                                     'hotel_system_transcript': hotel_conversation_turn_system_transcript})
        if turn_conversation_transcript:
            hotel_conversation_transcript.append(turn_conversation_transcript)
    # delete data not start at turn 0
    hotel_conversation_transcript_for_input = []
    for element in hotel_conversation_transcript:
        if element[0]['hotel_turn_id'] == '0':
            hotel_conversation_transcript_for_input.append(element)
    # the coversation pair will be [user_transcript[0], system[1]], except the last user_transcript with []
    hotel_conversation_pair = []
    for element in hotel_conversation_transcript_for_input:
        if len(element) > 1:
            for turn in range(len(element)):
                if turn != (len(element) - 1):
                    hotel_turn_conversation_pair = {"hotel_turn_id": turn,
                                                    "hotel_user_transcript": element[turn]['hotel_user_transcript'],
                                                    "hotel_system_transcript": element[turn + 1]['hotel_system_transcript']}
                    hotel_conversation_pair.append(hotel_turn_conversation_pair)
                else:
                    hotel_turn_conversation_pair = {"hotel_turn_id": turn,
                                                   "hotel_user_transcript": element[turn]['hotel_user_transcript'],
                                                    "hotel_system_transcript": ''}
                    hotel_conversation_pair.append(hotel_turn_conversation_pair)

    # for restaurant data collection
    restaurant_conversation_transcript = []
    for data in json_data_dic:
        if len(data['domains']) != 1:   # clean data not in the same region
            continue

        turn_conversation_transcript = []
        for turn in data['dialogue']:
            if turn['domain'] == "restaurant":
                turn_id = turn['turn_idx']
                restaurant_conversation_turn_transcript = turn['transcript']
                restaurant_conversation_turn_system_transcript = turn['system_transcript']
                turn_conversation_transcript.append({'restaurant_turn_id': str(turn_id),
                                                     'restaurant_user_transcript': restaurant_conversation_turn_transcript,
                                                     'restaurant_system_transcript': restaurant_conversation_turn_system_transcript})
        if turn_conversation_transcript:
            restaurant_conversation_transcript.append(turn_conversation_transcript)
    # delete data not start at turn 0
    restaurant_conversation_transcript_for_input = []
    for element in restaurant_conversation_transcript:
        if element[0]['restaurant_turn_id'] == '0':
            restaurant_conversation_transcript_for_input.append(element)
    # the coversation pair will be [user_transcript[0], system[1]], except the last user_transcript with []
    restaurant_conversation_pair = []
    for element in restaurant_conversation_transcript_for_input:
        if len(element) > 1:
            for turn in range(len(element)):
                if turn != (len(element) - 1):
                    restaurant_turn_conversation_pair = {"restaurant_turn_id": turn,
                                                    "restaurant_user_transcript": element[turn]['restaurant_user_transcript'],
                                                    "restaurant_system_transcript": element[turn + 1]['restaurant_system_transcript']}
                    restaurant_conversation_pair.append(restaurant_turn_conversation_pair)
                else:
                    restaurant_turn_conversation_pair = {"restaurant_turn_id": turn,
                                                   "restaurant_user_transcript": element[turn]['restaurant_user_transcript'],
                                                    "restaurant_system_transcript": ''}
                    restaurant_conversation_pair.append(restaurant_turn_conversation_pair)
    return [hotel_conversation_pair, restaurant_conversation_pair]




# dataset for text generation
def collect_data_for_classification():
    # collecting data
    data_from_j = open('dataset_for_domain.json', 'r')
    json_data_dic = json.loads(data_from_j.read())
    data_from_j.close()

    dialogue = []
    for data in json_data_dic:
        dialogue.append(data['dialogue'])

    transcript_label_pair = []
    for conversation in dialogue:
        for turn in conversation:
            transcript_label_pair.append({'transcript': turn['transcript'], 'turn_label': turn['turn_label']})

    price_range, name, time, food, ignore_word = [], [], [], [], []
    categories_message_with_label = []
    for pair in transcript_label_pair:
        labeled_transcript = pair['transcript']
        label_list = pair['turn_label']
        categories_turn_label = [0, 0, 0, 0, 0] # price, name, time, food, useless
        for label in label_list:
            if label[0] == 'restaurant-pricerange' or label[0] == 'hotel-pricerange':
                price_range.append({'transcript': labeled_transcript, 'label': label[1], 'key_label': [1, 0, 0, 0, 0]})
                categories_turn_label[0] = 1
            elif label[0] == 'hotel-name' or label[0] == 'restaurant-name' or label[0] == 'taxi-departure'\
                    or label[0] == 'taxi-departure' or label[0] == 'taxi-destination' or label[0] == 'attraction-name'\
                    or label[0] == 'train-departure' or label[0] == 'train-destination':
                name.append({'transcript': labeled_transcript, 'label': label[1], 'key_label': [0, 1, 0, 0, 0]})
                categories_turn_label[1] = 1
            elif label[0] == 'restaurant-book time' or label[0] == 'train-arriveby' or label[0] == 'train-leaveat'\
                    or label[0] == 'taxi-arriveby' or label[0] == 'taxi-leaveat':
                time.append({'transcript': labeled_transcript, 'label': label[1], 'key_label': [0, 0, 1, 0, 0]})
                categories_turn_label[2] = 1
            elif label[0] == 'restaurant-food':
                food.append({'transcript': labeled_transcript, 'label': label[1], 'key_label': [0, 0, 0, 1, 0]})
                categories_turn_label[3] = 1
        categories_message_with_label.append({'transcript': labeled_transcript, 'categories_label': categories_turn_label})

    # train categories model
    # preprocessing
    print("Build categories_model")
    message_without_empty_label = []
    for pair in categories_message_with_label:
        if pair['categories_label'] != [0, 0, 0, 0]:
            message_without_empty_label.append(pair)

    categories_train = message_without_empty_label[4000:]
    categories_test = message_without_empty_label[:4000]
    categories_train_message, categories_train_label = [], []
    for categories_pair in categories_train:
        categories_train_message.append(categories_pair['transcript'])
        categories_train_label.append(categories_pair['categories_label'])
    categories_test_message, categories_test_label = [], []
    for categories_pair in categories_test:
        categories_test_message.append(categories_pair['transcript'])
        categories_test_label.append(categories_pair['categories_label'])
    # Dictionary    !!! Tokenizer !!!
    tokenizer, padded_tokenized_message_train, padded_tokenized_message_test = \
        tokenization.tokenize_message(categories_train_message, categories_test_message)
    categories_train_dataset = [padded_tokenized_message_train, categories_train_label]
    categories_test_dataset = [padded_tokenized_message_test, categories_test_label]
    # train model part
    return tokenizer


def collect_tokenizer_for_user():
    # Preprocessing
    text_generation_dataset = collect_conversation()    # collect dataset
    dataset_for_hotel = text_generation_dataset[0]
    dataset_for_restaurant = text_generation_dataset[1]

    # data collect and setting part
    # label the message need to response or not
    # if system response is '', then the label will be 0 (means no need to response)
    user_input_message, system_output_message = [], []
    user_input_response_label, turn_stage_with_messages = [], []
    for conversation in dataset_for_hotel:
        user_input_message.append(conversation['hotel_user_transcript'])
        system_output_message.append(conversation['hotel_system_transcript'])
        turn_stage_with_messages.append({"turn_id": conversation['hotel_turn_id'],
                                         "user_input": conversation['hotel_user_transcript'],
                                         "system_output": conversation['hotel_system_transcript']
                                         })
        if conversation['hotel_system_transcript'] == '':
            user_input_response_label.append(0)     # 0 means no need to response
        else:
            user_input_response_label.append(1)     # 1 means need to response
    for conversation in dataset_for_restaurant:
        user_input_message.append(conversation['restaurant_user_transcript'])
        system_output_message.append(conversation['restaurant_system_transcript'])
        turn_stage_with_messages.append({"turn_id": conversation['restaurant_turn_id'],
                                         "user_input": conversation['restaurant_user_transcript'],
                                         "system_output": conversation['restaurant_system_transcript']
                                         })
        if conversation['restaurant_system_transcript'] == '':
            user_input_response_label.append(0)     # 0 means no need to response
        else:
            user_input_response_label.append(1)     # 1 means need to response


    # split data for train and test (train response or not)
    user_input_response_train = user_input_message[:8000]
    user_input_response_train_label = user_input_response_label[:8000]
    user_input_response_test = user_input_message[8000:]
    user_input_response_test_label = user_input_response_label[8000:]

    # Dictionary    !!! Tokenizer !!!
    tokenizer_for_response, padded_tokenized_user_response_train, padded_tokenized_user_response_test \
        = tokenization.tokenize_message(user_input_response_train, user_input_response_test)
    return tokenizer_for_response


def collect_tokenizer_for_system():
    # Preprocessing
    text_generation_dataset = collect_conversation()    # collect dataset
    dataset_for_hotel = text_generation_dataset[0]
    dataset_for_restaurant = text_generation_dataset[1]

    # data collect and setting part
    # label the message need to response or not
    # if system response is '', then the label will be 0 (means no need to response)
    user_input_message, system_output_message = [], []
    user_input_response_label, turn_stage_with_messages = [], []
    for conversation in dataset_for_hotel:
        user_input_message.append(conversation['hotel_user_transcript'])
        system_output_message.append(conversation['hotel_system_transcript'])
        turn_stage_with_messages.append({"turn_id": conversation['hotel_turn_id'],
                                         "user_input": conversation['hotel_user_transcript'],
                                         "system_output": conversation['hotel_system_transcript']
                                         })
        if conversation['hotel_system_transcript'] == '':
            user_input_response_label.append(0)     # 0 means no need to response
        else:
            user_input_response_label.append(1)     # 1 means need to response
    for conversation in dataset_for_restaurant:
        user_input_message.append(conversation['restaurant_user_transcript'])
        system_output_message.append(conversation['restaurant_system_transcript'])
        turn_stage_with_messages.append({"turn_id": conversation['restaurant_turn_id'],
                                         "user_input": conversation['restaurant_user_transcript'],
                                         "system_output": conversation['restaurant_system_transcript']
                                         })
        if conversation['restaurant_system_transcript'] == '':
            user_input_response_label.append(0)     # 0 means no need to response
        else:
            user_input_response_label.append(1)     # 1 means need to response


    # split data for train and test (train response or not)
    user_input_response_train = user_input_message[:8000]
    user_input_response_train_label = user_input_response_label[:8000]
    user_input_response_test = user_input_message[8000:]
    user_input_response_test_label = user_input_response_label[8000:]

    # Dictionary    !!! Tokenizer !!!
    tokenizer_for_response, padded_tokenized_user_response_train, padded_tokenized_user_response_test \
        = tokenization.tokenize_message(user_input_response_train, user_input_response_test)
    # train model
    response_train = [padded_tokenized_user_response_train, user_input_response_train_label]
    response_test = [padded_tokenized_user_response_test, user_input_response_test_label]


    # stage check model (using functional structure)
    stage_user_message, stage_system_message, stage_label = [], [], []
    for turn in turn_stage_with_messages:
        stage_user_message.append(turn['user_input'])
        stage_system_message.append(turn['system_output'])
        if turn['turn_id'] == 0:
            turn_stage_label = [1, 0, 0]
        elif turn['system_output'] == '':
            turn_stage_label = [0, 0, 1]
        else:
            turn_stage_label = [0, 1, 0]
        stage_label.append(turn_stage_label)

    user_input_stage_train = stage_user_message[:8000]
    user_input_stage_train_label = stage_label[:8000]
    user_input_stage_test = stage_user_message[8000:]
    user_input_stage_test_label = stage_label[8000:]

    # add 'start=(index3)' and 'end'=(index2) in the corresponding position
    stage_system_message_with_tag = []
    for sentence in stage_system_message:
        if sentence != '':
            stage_system_message_with_tag.append('start ' + sentence + ' end') # no /start, /end since not in all sentence
    system_input_stage_train = stage_system_message_with_tag[:6000]
    system_input_stage_test = stage_system_message_with_tag[6000:]
    
    # 2 Dictionary
    # user dictionary
    tokenizer_for_stage_for_system, padded_tokenized_system_stage_train, padded_tokenized_system_stage_test \
        = tokenization.tokenize_message_for_response(system_input_stage_train, system_input_stage_test)
    return tokenizer_for_stage_for_system

