import os
import torch
from transformers import BertTokenizer, EncoderDecoderModel
import csv
from datetime import datetime

# Path to your dialogues file and model
file_path = 'dialogs.txt'
model_path = './results/final_model'
current_datetime = datetime.now().strftime("%m-%d %H-%M-%S")
current_datetime = str(current_datetime)

def preprocess_dialogues(file_path):
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                dialogues.append((parts[0], parts[1]))
    return dialogues

# Preprocess the dialogues
dialogues = preprocess_dialogues(file_path)

def generate_response(model, tokenizer, context, device):
    # Ensure context is encoded correctly
    input_ids = tokenizer.encode_plus(context, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)['input_ids']
    
    # Specify decoder_start_token_id if it's not set in model config
    decoder_start_token_id = model.config.decoder_start_token_id if model.config.decoder_start_token_id is not None else tokenizer.cls_token_id

    # Generate the output
    output_ids = model.generate(
        input_ids, 
        decoder_start_token_id=decoder_start_token_id, 
        max_length=512, 
        num_beams=5, 
        early_stopping=True, 
        temperature=0.7, 
        do_sample=True
    )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Load the BERT tokenizer and model as before
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = './results/checkpoint-epoch-2'
model = EncoderDecoderModel.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Open a CSV file to write the results
with open('test_results_'+current_datetime+'.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Context', 'Expected Response', 'Generated Response'])

    # Test the model on each dialogue
    for context, expected_response in dialogues:
        generated_response = generate_response(model, tokenizer, context, device)
        writer.writerow([context, expected_response, generated_response])
        print(f'Context: {context}')
        print(f'Expected Response: {expected_response}')
        print(f'Generated Response: {generated_response}\n')

print("Testing complete. Results saved to test_results.csv")
    
