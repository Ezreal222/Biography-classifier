import re
import sys
import math

filename = sys.argv[1]
N = int(sys.argv[2])


def process_data(text):
    entries = re.split(r'\n{2,}', text.strip())
    processed_entries = []

    for entry in entries:
        lines = entry.split('\n')
        # Keep periods within words
        name = re.findall(r'[\w.]+', lines[0].lower().strip())
        category = [lines[1].lower().strip()]
        # Keep periods within words
        bio = re.findall(r'[\w.]+', ' '.join(lines[2:]).lower().strip())

        processed_entries.append([name, category, bio])

    return processed_entries


with open(filename, 'r') as file:
    text = file.read()

formatted_bio = process_data(text)
training_set = formatted_bio[:N]
test_set = formatted_bio[N:]

# helper function to normalize the data


def preprocess_data(data):
    with open("stopwords.txt", "r") as f:
        stopwords = f.read().split()

    preprocessed_data = []

    for biography in data:
        preprocessed_biography = []
        for sentence in biography:
            preprocessed_sentence = [word.lower() for word in sentence]
            preprocessed_sentence = [
                word for word in preprocessed_sentence if word not in stopwords]
            preprocessed_sentence = [word.strip(
                ",.!?") for word in preprocessed_sentence]
            preprocessed_sentence = [
                word for word in preprocessed_sentence if len(word) > 2]
            preprocessed_biography.append(preprocessed_sentence)
        preprocessed_data.append(preprocessed_biography)

    return preprocessed_data

# Learning the classifier from the training set


# 1. Normalization of the training set
training_set = preprocess_data(training_set)

# 2. Counting the number of each category in the training set
categories = {}
for i in range(len(training_set)):
    if training_set[i][1][0] not in categories:
        categories[training_set[i][1][0]] = 1
    else:
        categories[training_set[i][1][0]] += 1

# 3. Probabilities

# extract all the words in the training set
all_words = []
for biography in training_set:
    for word in biography[2]:
        if word not in all_words:
            all_words.append(word)

categories_words = {}
for word in all_words:
    categories_words[word] = {}
    for category in categories:
        categories_words[word][category] = 0
    for biography in training_set:
        if word in biography[2]:
            categories_words[word][biography[1][0]] += 1

categories_freq = {}
for key, value in categories.items():
    categories_freq[key] = value / len(training_set)


# P(C)
P_C = {}
for key, value in categories_freq.items():
    P_C[key] = (value + 0.1) / (1 + 0.1 * len(categories_freq))

# -log(P(C))
log_P_C = {}
for key, value in P_C.items():
    log_P_C[key] = round(-math.log(value, 2), 4)


# L(W|C)
# print(categories_words)
L_W_C = {}
for key, value in categories_words.items():
    L_W_C[key] = {}
    for category, count in value.items():
        L_W_C[key][category] = round(
            -math.log((count / categories[category] + 0.1) / (1 + 0.1 * 2), 2), 4)


# Testing the classifier on the test set

# 1. Normalization of the test set
test_set = preprocess_data(test_set)

# Filter the test set to only contain words that are present in all_words
filtered_test_set = []
for biography in test_set:
    filtered_biography = []
    for word in biography[2]:
        if word in all_words:
            filtered_biography.append(word)
    filtered_test_set.append([biography[0], biography[1], filtered_biography])

# L(C|B)
L_C_B = {}
for biography in filtered_test_set:
    L_C_B[biography[0][1]] = {}
    for key, value in log_P_C.items():
        L_C_B[biography[0][1]][key] = value
        for word in biography[2]:
            L_C_B[biography[0][1]][key] += L_W_C[word][key]

for key, value in L_C_B.items():
    for key2, value2 in value.items():
        L_C_B[key][key2] = round(L_C_B[key][key2], 4)


def calculate_probabilities(L_C_B):
    output = {}
    for name, category_values in L_C_B.items():
        c_i = list(category_values.values())
        m = min(c_i)

        x_i = []
        for value in c_i:
            if value - m < 7:
                x_i.append(2 ** (m - value))
            else:
                x_i.append(0)

        s = sum(x_i)
        probabilities = {category: x / s for category,
                         x in zip(category_values.keys(), x_i)}
        output[name] = probabilities

    return output


probabilities = calculate_probabilities(L_C_B)

# output
correct_predictions = 0
total_predictions = len(filtered_test_set)

for biography, category_probabilities in zip(test_set, probabilities.values()):
    name = ' '.join(biography[0]).capitalize()
    correct_category = biography[1][0]
    predicted_category = max(category_probabilities,
                             key=category_probabilities.get)
    is_correct = predicted_category == correct_category
    correct_predictions += is_correct

    print(f"{name}. Prediction: {predicted_category.capitalize()}. {'Right' if is_correct else 'Wrong'}.")
    for category, probability in category_probabilities.items():
        print(f"{category.capitalize()}: {probability:.2f}", end=' ')
    print('\n')

accuracy = correct_predictions / total_predictions
print(
    f"Overall accuracy: {correct_predictions} out of {total_predictions} = {accuracy:.2f}.")
