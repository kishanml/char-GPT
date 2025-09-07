import os
import torch
import torch.optim as optim
from core import charGPT,DataManager

BATCH_SIZE = 128
BLOCK_SIZE = 256
EMBEDDING_DIM = 300
LEARNING_RATE = 5e-4
MAX_ITERS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
BEST_VAL_LOSS = float('inf')
MODEL_SAVE_PATH = "best_model.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "input_2.txt"

@torch.no_grad()
def estimate_loss(model, data_manager, device):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = data_manager.get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


    print(f"Starting training on {device}...")
    for iter_num in range(MAX_ITERS):

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, data_manager, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

            prompt = "Life is"
            context = torch.tensor(data_manager.encoder(prompt)).view(1,-1).to(device)
            op_text = data_manager.decoder(model.generate(context, max_new_tokens=100,device=device)[0].tolist())
            print(op_text)
            with open(f'results/output_{iter_num+1}.txt','w') as f:
                f.write(op_text)

            if losses['test'] < BEST_VAL_LOSS:
                BEST_VAL_LOSS = losses['test']
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Validation loss improved! Saving model to {MODEL_SAVE_PATH}")

        xb, yb = data_manager.get_batch('train', device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

   