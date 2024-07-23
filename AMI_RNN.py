import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data from provided URLs
train_df = pd.read_csv('https://raw.githubusercontent.com/apilny2/AMI/main/ami_train.csv')
val_df = pd.read_csv('https://raw.githubusercontent.com/apilny2/AMI/main/ami_validation.csv')
test_df = pd.read_csv('https://raw.githubusercontent.com/apilny2/AMI/main/ami_test.csv')

# Combine labels from train, validation, and test datasets
all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])

# Initialize Label Encoder and fit on all possible labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform labels in each dataset
train_df['label'] = label_encoder.transform(train_df['label'])
val_df['label'] = label_encoder.transform(val_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Optionally, remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])

max_length = max([len(s.split()) for s in train_df['text']])

def prepare_data(df):
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded, df['label']

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

# RNN Model With Bidirectional LSTM
vocab_size = len(tokenizer.word_index) + 1
output_size = y_train.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Bidirectional(LSTM(128, return_sequences=True)))  # First Bidirectional LSTM layer
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))  # Second Bidirectional LSTM layer
model.add(Dropout(0.5))
model.add(Dense(output_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=6, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print('Test Set Accuracy:', accuracy)

# Unique labels across all data splits
unique_labels = np.unique(np.concatenate([train_df['label'], val_df['label'], test_df['label']]))

# Ensuring class names match the unique labels and are strings
class_names = [str(label) for label in label_encoder.inverse_transform(unique_labels)]

# Convert y_test from categorical to label encoding
y_test_labels = np.argmax(y_test_categorical, axis=1)

# Generate predictions
predictions = model.predict(X_test)
flat_predictions = np.argmax(predictions, axis=1)

# Classification Report with zero_division parameter
classificationReport = classification_report(
    y_test_labels, flat_predictions, labels=unique_labels,
    target_names=class_names, zero_division=1
)
print('Classification Report:\n', classificationReport)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test_labels, flat_predictions)
print('Matthews Correlation Coefficient:', mcc)

# Calculate Micro F1-Score
micro_f1 = f1_score(y_test_labels, flat_predictions, average='micro')
print('Micro F1-Score:', micro_f1)

# Calculate and print Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, flat_predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
print("Confusion Matrix:\n", conf_matrix_df)
