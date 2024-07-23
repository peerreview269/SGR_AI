import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Check and print GPU information
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'GPU Name: {torch.cuda.get_device_name(device)}')
else:
    device = torch.device('cpu')
    print('No GPU available, using CPU instead.')

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the AMI datasets
train_df = pd.read_csv('https://raw.githubusercontent.com/peerreview269/SGR_AI/main/ami_train.csv')
val_df = pd.read_csv('https://raw.githubusercontent.com/peerreview269/SGR_AI/main/ami_validation.csv')
test_df = pd.read_csv('https://raw.githubusercontent.com/peerreview269/SGR_AI/main/ami_test.csv')

# Combine labels from train, validation, and test datasets
all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])

# Initialize Label Encoder and fit on all possible labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform labels in each dataset
train_df['label'] = label_encoder.transform(train_df['label'])
val_df['label'] = label_encoder.transform(val_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

num_classes = len(label_encoder.classes_)
y_test_categorical = np.eye(num_classes)[test_df['label']]

def encode_data_with_context(df, window_size, context_type):
    input_ids = []
    attention_masks = []

    for index, row in df.iterrows():
        current_dialogue_id = row['Name']
        context_string = ""

        # Add previous context
        if context_type in ['previous', 'both']:
            previous_context = ' '.join(df[(df['Name'] == current_dialogue_id) & (df.index < index)].tail(window_size)['text'])
            context_string += previous_context + " "

        # Add target utterance
        context_string += row['text'] + " "

        # Add subsequent context
        if context_type in ['subsequent', 'both']:
            subsequent_context = ' '.join(df[(df['Name'] == current_dialogue_id) & (df.index > index)].head(window_size)['text'])
            context_string += subsequent_context

        # Tokenize and encode
        encoded_dict = tokenizer.encode_plus(
            context_string,
            add_special_tokens=True,
            max_length=512,  # Adjust as needed
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'].values)  # Ensure this matches your label column name

    return input_ids, attention_masks, labels

# Define specific configurations
custom_configurations = [(0, 'both'), (1, 'previous'), (2, 'previous'), (1, 'subsequent'), (2, 'subsequent'), (1, 'both'), (2, 'both')]

# Iterate over custom configurations
for config in custom_configurations:
    window_size, context_type = config

    # Encoding data with the specified context
    train_inputs, train_masks, train_labels = encode_data_with_context(train_df, window_size, context_type)
    val_inputs, val_masks, val_labels = encode_data_with_context(val_df, window_size, context_type)
    test_inputs, test_masks, test_labels = encode_data_with_context(test_df, window_size, context_type)

    # Creating DataLoaders
    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Initialize the model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes,
    )
    model.to(device)

    # Optimizer and training parameters
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 6
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    total_start_time = time.time()

    # Training loop
    for epoch_i in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.zero_grad()

            with autocast():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            total_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Context: {context_type}, Window Size: {window_size}, Epoch: {epoch_i + 1}, Avg Training Loss: {avg_train_loss}")

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        minutes, seconds = divmod(int(total_training_time), 60)
        print(f"Total training time for Window Size: {window_size}, Context Type: {context_type}: {minutes} minutes, {seconds} seconds")

    # Test set evaluation
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()  # Set the model to evaluation mode
    predictions, true_labels = [], []

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    # Flatten the predictions and true labels
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Unique labels across all data splits
    unique_labels = np.unique(np.concatenate([train_df['label'], val_df['label'], test_df['label']]))
    
    # Ensuring class names match the unique labels
    class_names = [str(name) for name in label_encoder.inverse_transform(unique_labels)]

    # Convert y_test from categorical to label encoding
    y_test_labels = np.argmax(y_test_categorical, axis=1)

    # Classification Report with specified labels
    classificationReport = classification_report(
        y_test_labels, flat_predictions, labels=unique_labels, 
        target_names=class_names,zero_division=1)
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
