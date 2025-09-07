import os
import torch
import torch.optim as optim
from core import charGPT,DataManager

BATCH_SIZE = 128
BLOCK_SIZE = 256
EMBEDDING_DIM = 300
MODEL_SAVE_PATH = "best_model.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "input_2.txt"

if not os.path.exists(data_path):
    print(f"Error: The file '{data_path}' was not found. Please ensure it's in the same directory.")
else:
    data_manager = DataManager(data_source_path=data_path, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)

    model = charGPT(
        vocab_size=data_manager.vocab_size,
        block_size=data_manager.block_size,
        embedding_dim=EMBEDDING_DIM,
        no_of_attn_layers=6,
        no_of_heads=6
    )
    model = model.to(device)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    prompt = "Hey " 
    context = torch.tensor(data_manager.encoder(prompt)).view(1,-1).to(device)
    op_text = data_manager.decoder(model.generate(context, max_new_tokens=500,device=device)[0].tolist())
    # with open('output.txt','w') as f:
    #     f.write(op_text)
    print(op_text)
        
    
