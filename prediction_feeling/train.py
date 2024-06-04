import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle


def load_data(data_dir):
    texts, labels = [], []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
    return texts, labels


train_texts, train_labels = load_data('aclImdb/train')
test_texts, test_labels = load_data('aclImdb/test')

max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

x_train = tokenizer.texts_to_sequences(train_texts)
x_test = tokenizer.texts_to_sequences(test_texts)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
y_train = np.array(train_labels)
y_test = np.array(test_labels)


def build_model(model_type='RNN'):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=max_len))
    if model_type == 'RNN':
        model.add(SimpleRNN(32))
    elif model_type == 'LSTM':
        model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


models = ['RNN', 'LSTM']
results = []

for model_type in models:
    model = build_model(model_type)
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    results.append([model_type, accuracy])


    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'{model_type} Loss')
    plt.legend()
    plt.savefig(f'{model_type}_loss.png')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title(f'{model_type} Accuracy')
    plt.legend()
    plt.savefig(f'{model_type}_accuracy.png')

    plt.close()


results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
print(results_df)


best_model_type = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_model = build_model(best_model_type)
best_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
y_pred = (best_model.predict(x_test) > 0.5).astype("int32")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Best Model: {best_model_type}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n {conf_matrix}")


def show_examples(texts, y_true, y_pred, label, correct=True, num_examples=5):
    examples = []
    for text, true, pred in zip(texts, y_true, y_pred):
        if (true == label) and ((true == pred) == correct):
            examples.append(text)
        if len(examples) >= num_examples:
            break
    return examples


pos_correct = show_examples(test_texts, y_test, y_pred, label=1, correct=True)
neg_correct = show_examples(test_texts, y_test, y_pred, label=0, correct=True)
pos_incorrect = show_examples(test_texts, y_test, y_pred, label=1, correct=False)
neg_incorrect = show_examples(test_texts, y_test, y_pred, label=0, correct=False)


with open('prediction_examples.txt', 'w', encoding='utf-8') as f:
    f.write("Positive Correct Examples:\n")
    for example in pos_correct:
        f.write(example + "\n\n")
    
    f.write("Negative Correct Examples:\n")
    for example in neg_correct:
        f.write(example + "\n\n")
    
    f.write("Positive Incorrect Examples:\n")
    for example in pos_incorrect:
        f.write(example + "\n\n")
    
    f.write("Negative Incorrect Examples:\n")
    for example in neg_incorrect:
        f.write(example + "\n\n")

print("Positive Correct Examples:", pos_correct)
print("Negative Correct Examples:", neg_correct)
print("Positive Incorrect Examples:", pos_incorrect)
print("Negative Incorrect Examples:", neg_incorrect)


best_model.save('best_model.h5')


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Entrenamiento y evaluación completos.")
