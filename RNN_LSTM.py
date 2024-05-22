import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data
# Load silicone dataset
silicone_dataset = load_dataset('silicone', 'dyda_da')
train_df = pd.DataFrame(silicone_dataset['train'])
val_df = pd.DataFrame(silicone_dataset['validation'])
test_df = pd.DataFrame(silicone_dataset['test'])

# Combine labels from train, validation, and test datasets
all_labels = pd.concat([train_df['Label'], val_df['Label'], test_df['Label']])

# Initialize Label Encoder and fit on all possible labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform labels in each dataset
train_df['Label'] = label_encoder.transform(train_df['Label'])
val_df['Label'] = label_encoder.transform(val_df['Label'])
test_df['Label'] = label_encoder.transform(test_df['Label'])

# Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Optionally, remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_df['Utterance'] = train_df['Utterance'].apply(preprocess_text)
val_df['Utterance'] = val_df['Utterance'].apply(preprocess_text)
test_df['Utterance'] = test_df['Utterance'].apply(preprocess_text)

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['Utterance'])

max_length = max([len(s.split()) for s in train_df['Utterance']])

def prepare_data(df):
    sequences = tokenizer.texts_to_sequences(df['Utterance'])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded, df['Label']  # Change here from 'Encoded_Code' to 'Label'

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

## RNN Model With Bidirectional LSTM
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
unique_labels = np.unique(np.concatenate([train_df['Label'], val_df['Label'], test_df['Label']]))

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
