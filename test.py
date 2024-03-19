import torch
from transformers import BertTokenizer, EncoderDecoderModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention(attention, tokens, figsize=(10, 10)):
    # Convert to array
    attention = np.array(attention)

    # Cut off the padding (typically represented by zeros in the attention array)
    # This assumes that 'attention' is a square array with shape (seq_len, seq_len)
    seq_len = len(tokens)
    attention = attention[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    # Set up plot
    plt.figure(figsize=figsize)
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, square=True)
    plt.show()


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Specify the checkpoint you want to test
checkpoint_path = './results/checkpoint-epoch-2'  # adjust this to your checkpoint path

# Load the model from the checkpoint
model = EncoderDecoderModel.from_pretrained(checkpoint_path)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Test the model with a sample message
message = "will you let me drive the car today?"  # replace this with your test message

#encoding
input_ids = tokenizer.encode_plus(message, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)['input_ids']

# Perform a forward pass to get the attention weights
with torch.no_grad():
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, output_attentions=True)
    attentions = outputs.encoder_attentions[-1]  # Get the attention from the last layer of the encoder


# Decode the input for token labels
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Remove padding and get actual sequence length
seq_len = sum(input_ids[0] != tokenizer.pad_token_id)
tokens = tokens[:seq_len]
attentions = attentions[:, :, :seq_len, :seq_len]  # Remove padding

# Plot the attention maps
for head in range(attentions.size(1)):
    print(f'Head {head+1}:')
    attention_head = attentions[0, head].detach().cpu().numpy()  # 0 for grabbing the first example
    plot_attention(attention_head, tokens)


# Explicitly specify the decoder_start_token_id
decoder_start_token_id = tokenizer.cls_token_id

# Set the temperature parameter and enable sampling
temperature = 0.7  # Adjust as needed
do_sample = True  # Enable sampling

# Generate the output
output_ids = model.generate(input_ids,
                            decoder_start_token_id=decoder_start_token_id,
                            max_length=512,
                            num_beams=5, 
                            early_stopping=True,
                            temperature = temperature,
                            do_sample=do_sample)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f'Input: {message}')
print(f'Output: {output}')
