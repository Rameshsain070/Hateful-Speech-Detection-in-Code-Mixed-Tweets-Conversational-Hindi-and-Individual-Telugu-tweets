import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import joblib

# Loading the CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Droping unnecessary columns if present
train_df = train_df[['S.No', 'Comments', 'Label']]
test_df = test_df[['S.No', 'Comments', 'Label']]

# Printing the column names to verify
print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)

# Encoding labels
label_encoder = LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
test_df['Label'] = label_encoder.transform(test_df['Label'])

# Verifying the number of unique labels
num_labels = len(label_encoder.classes_)
print(f"Number of unique labels: {num_labels}")

# Defining dataset class
class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        comment = str(self.dataframe.iloc[index]['Comments'])
        label = self.dataframe.iloc[index]['Label']
        
        # Tokenizing the text
        encoding = self.tokenizer(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Loading DistilBERT tokenizer and model
model_name = 'distilbert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Preparing datasets
train_dataset = TweetDataset(train_df, tokenizer, max_len=128)
test_dataset = TweetDataset(test_df, tokenizer, max_len=128)

# Defining custom evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model='f1',  
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()

# Saving the trained model and tokenizer
trainer.save_model('./distilbert_model')  
tokenizer.save_pretrained('./distilbert_model')  

# Saving the model using joblib for later use
joblib.dump(model, './distilbert_model/distilbert_trained_model.joblib')
print("Model saved successfully using joblib.")
