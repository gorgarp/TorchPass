import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import time
import os
import multiprocessing
import warnings

# Comment the line below if you want warnings shown.
# Some dependencies haven't yet updated torch.amp, which creates warnings.
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up CUDA if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine number of available CPU cores and use <80% of them
available_cores = multiprocessing.cpu_count()
used_cores = max(1, int(available_cores * 0.8))  # At least 1 core
torch.set_num_threads(used_cores)

# Print the device being used, but only in the main process
if multiprocessing.current_process().name == "MainProcess":
    print(f"Using device: {DEVICE}")

# Function to initialize workers and suppress their output
def init_worker(worker_id):
    import sys
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# Custom dataset class for handling password data
class PassData(Dataset):
    def __init__(self, passwords, max_len):
        # Initialize dataset with passwords and maximum length
        self.passwords = passwords
        self.max_len = max_len
        
        # Create character to index mapping
        self.char_to_idx = {chr(i): i - 31 for i in range(32, 127)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<START>'] = len(self.char_to_idx)
        self.char_to_idx['<END>'] = len(self.char_to_idx)
        
        # Create reverse mapping (index to character)
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        
        # Set vocabulary size
        self.vocab_size = len(self.char_to_idx)

    def __len__(self):
        # Return the number of passwords in the dataset
        return len(self.passwords)

    def __getitem__(self, idx):
        # Encode a password into a tensor of indices
        password = self.passwords[idx]
        encoded = [self.char_to_idx['<START>']]
        for c in password:
            if c in self.char_to_idx:
                encoded.append(self.char_to_idx[c])
        encoded.append(self.char_to_idx['<END>'])
        
        # Pad the encoded password to the maximum length
        encoded += [self.char_to_idx['<PAD>']] * (self.max_len - len(encoded))
        
        return torch.tensor(encoded[:self.max_len], dtype=torch.long)

# Neural network model for password generation
class PassGen(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(PassGen, self).__init__()
        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convert input indices to embeddings
        x = self.embedding(x)
        
        # Process through LSTM
        x, _ = self.lstm(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Generate logits through fully connected layer
        return self.fc(x)

# Function to train the model
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    model.train()
    best_val_loss = float('inf')
    no_improve = 0
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(num_epochs):
        start_time = time.time()
        total_train_loss = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use automatic mixed precision for faster training
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
            
            # Scale gradients and perform backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Use automatic mixed precision for faster training
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Check for improvement and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), 'best_model.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        model.train()
    
    # Load the best model
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    else:
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    
    return model

    
    # Load the best model
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    else:
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    
    return model

# Function to generate a batch of passwords
def gen_pass_batch(model, char_to_idx, idx_to_char, batch_size, min_len=8, max_len=26, temp=1.0):
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        current_chars = torch.tensor([[char_to_idx['<START>']] for _ in range(batch_size)], dtype=torch.long).to(DEVICE)
        passwords = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            outputs = model(current_chars)
            outputs = outputs[:, -1, :] / temp
            probs = torch.softmax(outputs, dim=-1)
            next_chars = torch.multinomial(probs, 1).squeeze(1)

            for i, next_char in enumerate(next_chars):
                if next_char == char_to_idx['<END>'] and len(passwords[i]) >= min_len:
                    finished[i] = True
                elif next_char != char_to_idx['<PAD>'] and not finished[i]:
                    passwords[i].append(idx_to_char[next_char.item()])

            if all(finished):
                break
            
            current_chars = torch.cat([current_chars, next_chars.unsqueeze(1)], dim=1)

        # Return generated passwords, ignoring those that didn't meet the minimum length requirement
        return [''.join(pwd) for pwd in passwords if len(pwd) >= min_len]

# Main function to handle command-line arguments and run the program
def main():
    parser = argparse.ArgumentParser(description="AI Password Generator")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--input', type=str, help="Input file for training")
    parser.add_argument('--output', type=str, help="Output file for generated passwords")
    parser.add_argument('--model', type=str, default='model.pth', help="Model file path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=256, help="Batch size")
    parser.add_argument('--num_pass', type=int, default=100, help="Number of passwords to generate")
    parser.add_argument('--temp', type=float, default=1.0, help="Temperature for generation")
    parser.add_argument('--workers', type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.input:
            raise ValueError("Input file required for training")
        
        if os.path.exists(args.model):
            print(f"Loading existing model: {args.model}")
            checkpoint = torch.load(args.model, map_location=DEVICE)
            char_to_idx = checkpoint['char_to_idx']
            idx_to_char = checkpoint['idx_to_char']
            model = PassGen(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Existing model loaded successfully. Continuing training.")
        else:
            print("No existing model found. Initializing a new model.")
            char_to_idx = {chr(i): i - 31 for i in range(32, 127)}
            char_to_idx['<PAD>'] = 0
            char_to_idx['<START>'] = len(char_to_idx)
            char_to_idx['<END>'] = len(char_to_idx)
            idx_to_char = {i: char for char, i in char_to_idx.items()}
            model = PassGen(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(DEVICE)

        # Load and preprocess password data
        with open(args.input, 'r', encoding='latin-1') as f:
            passwords = [line.strip() for line in f if 8 <= len(line.strip()) <= 26 and all(32 <= ord(c) < 127 for c in line.strip())]

        random.shuffle(passwords)
        max_len = 28
        train_passwords = passwords[:int(0.9 * len(passwords))]
        val_passwords = passwords[int(0.9 * len(passwords)):]

        train_dataset = PassData(train_passwords, max_len)
        val_dataset = PassData(val_passwords, max_len)

        train_dataset.char_to_idx = char_to_idx
        train_dataset.idx_to_char = idx_to_char
        train_dataset.vocab_size = len(char_to_idx)
        val_dataset.char_to_idx = char_to_idx
        val_dataset.idx_to_char = idx_to_char
        val_dataset.vocab_size = len(char_to_idx)

        # Set up loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=init_worker)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=init_worker)

        print("Starting training...")
        model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs)

        # Save the trained model
        if torch.cuda.device_count() > 1:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, args.model)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, args.model)
        print(f"Model saved: {args.model}")

    elif args.mode == 'generate':
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        if not args.output:
            raise ValueError("Output file required for generation")

        # Load the model
        print(f"Loading model: {args.model}")
        checkpoint = torch.load(args.model, map_location=DEVICE)
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        model = PassGen(len(char_to_idx), embed_size=256, hidden_size=512, num_layers=3)
        model.load_state_dict(checkpoint['model_state_dict'])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(DEVICE)
        model.eval()

        # Generate passwords in batches
        print(f"Generating passwords...")
        with open(args.output, 'w', encoding='utf-8') as f:
            total_generated = 0
            pbar = tqdm(total=args.num_pass, desc="Generating passwords")
            while total_generated < args.num_pass:
                current_batch_size = min(args.num_pass - total_generated, args.batch)
                batch_passwords = gen_pass_batch(model, char_to_idx, idx_to_char, current_batch_size, temp=args.temp)
                for password in batch_passwords:
                    f.write(f"{password}\n")
                new_passwords = len(batch_passwords)
                total_generated += new_passwords
                pbar.update(new_passwords)
                print(f"Generated {total_generated}/{args.num_pass} passwords")
            pbar.close()

        print(f"Passwords saved to: {args.output}")

if __name__ == "__main__":
    main()
