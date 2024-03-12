import tensorflow as tf
from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer, AdamW
from torch.utils.data import Dataset
import torch
import glob
import os

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.cuda.empty_cache()

# Define your custom dataset
class ChatDataset(Dataset):
    def __init__(self, tfrecord_files):
        self.examples = []
        for tfrecord_file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            for raw_record in raw_dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                context = [value.decode('utf-8') for value in example.features.feature['context'].bytes_list.value]
                response = example.features.feature['response'].bytes_list.value[0].decode('utf-8')
                self.examples.append((context, response))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, response = self.examples[idx]
        # Join all context messages into a single string with [SEP] tokens in between
        context = ' [SEP] '.join(context)
        context_encoding = tokenizer.encode_plus(context, truncation=True, padding='max_length', max_length=512)
        response_encoding = tokenizer.encode_plus(response, truncation=True, padding='max_length', max_length=512)
        return {
            'input_ids': torch.tensor(context_encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(context_encoding['attention_mask'], dtype=torch.long),
            #'token_type_ids': torch.tensor(context_encoding['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(response_encoding['input_ids'], dtype=torch.long)
        }

# Load your data
# Get a list of all TFRecord files
tfrecord_files = glob.glob('dataset/dataflow_output/train-*.tfrecord')
dataset = ChatDataset(tfrecord_files)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if there are any checkpoints
checkpoints = [dir for dir in os.listdir(training_args.output_dir) if dir.startswith('checkpoint')]

if checkpoints:
    # If there are checkpoints, get the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    print(f"Continuing training from checkpoint {latest_checkpoint}")
    
    # Load the model from the latest checkpoint
    model = EncoderDecoderModel.from_pretrained(f"{training_args.output_dir}/{latest_checkpoint}")
else:
    # If there are no checkpoints, initialize a new model
    print("No checkpoints found. Starting training from scratch.")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Specify the special tokens
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# Define the gradient accumulation steps
gradient_accumulation_steps = 4 

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    fp16=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# for i in range(5):
#     example = dataset[i]
#     print(f'Example {i}:')
#     print(f'Input IDs: {example["input_ids"]}')
#     print(f'Attention Mask: {example["attention_mask"]}')
#     # print(f'Token Type IDs: {example["token_type_ids"]}')
#     print(f'Labels: {example["labels"]}')
#     print()

# # Test Model Forward Pass
# print("Testing Model Forward Pass...")
# example = dataset[0]
# outputs = model(
#     input_ids=example['input_ids'].unsqueeze(0).to(device), 
#     attention_mask=example['attention_mask'].unsqueeze(0).to(device), 
#     # token_type_ids=example['token_type_ids'].unsqueeze(0).to(device),
#     decoder_input_ids=example['labels'].unsqueeze(0).to(device)
# )
# print(outputs)

# Train the model
print("Starting Training...")
save_steps = 5000  # adjust this value to fit your needs
log_steps = 100  # adjust this value to fit your needs
total_loss = 0.0  # to accumulate loss for logging

for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(trainer.get_train_dataloader()):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        
        # Compute loss
        loss = outputs.loss / gradient_accumulation_steps  # Normalize the loss
        total_loss += loss.item()  # Accumulate the loss
        
        loss.backward()
        
        # Perform an optimizer step and zero the gradients every `gradient_accumulation_steps` steps
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Log the loss every `log_steps` steps
        if step % log_steps == 0 and step > 0:
            avg_loss = total_loss / log_steps
            print(f"Step: {step}, Average Loss: {avg_loss}")
            total_loss = 0.0  # Reset the total loss

        # Save a checkpoint every `save_steps` steps
        if step % save_steps == 0:
            trainer.save_model(f"{training_args.output_dir}/checkpoint-{step}")
