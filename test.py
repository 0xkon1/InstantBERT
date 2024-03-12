import torch
import os
from transformers import BertTokenizer, EncoderDecoderModel, BertConfig, EncoderDecoderConfig

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Specify the checkpoint you want to test
checkpoint_path = './results/checkpoint-35000'  # adjust this to your checkpoint path

# Load the configuration from the checkpoint
config_path = os.path.join(checkpoint_path, "config.json")
config = BertConfig.from_pretrained(config_path)

# Load the generation config
generation_config_path = os.path.join(checkpoint_path, "generation_config.json")
generation_config = BertConfig.from_pretrained(generation_config_path)

# Create an EncoderDecoderConfig
encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, generation_config)

# Load the model from the checkpoint using the configuration
model = EncoderDecoderModel.from_pretrained(checkpoint_path, config=encoder_decoder_config)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Specify the special tokens after moving the model to the device
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# Test the model with a sample message
message = "Hey, wanna go out today?"  # replace this with your test message
input_ids = tokenizer.encode(message, return_tensors='pt').to(device)
output_ids = model.generate(input_ids, decoder_start_token_id=tokenizer.cls_token_id, bos_token_id=tokenizer.cls_token_id, max_new_tokens=50, temperature=0.8)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f'Input: {message}')
print(f'Output: {output}')
