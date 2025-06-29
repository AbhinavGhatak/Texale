import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import pickle
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

print("🚀 Starting Mini-LLM Training on Harry Potter Text...")

""" Phase 1: Load Dataset and Preprocessing """
try:
    with open("texcore.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("❌ texcore.txt not found! Please make sure your Harry Potter text file is named 'texcore.txt'")
    exit(1)

# Print sample to verify loading
print(f"📚 Dataset loaded successfully!")
print(f"📖 Sample text: {text[:200]}...")
print(f"📊 Total characters: {len(text):,}")

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 Vocabulary size: {vocab_size}")
print(f"🔤 Characters: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")

# Character mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l if i in itos])


""" Phase 2: Data Preparation and Batching """
# Hyperparameters - Optimized for better training
block_size = 256      # Longer context for better understanding
batch_size = 32       # Reduced for memory efficiency but good training
n_embd = 384         # Larger embedding dimension
n_head = 6           # Number of attention heads
n_layer = 6          # Number of transformer blocks
dropout = 0.2        # Dropout for regularization
max_iters = 10000    # More training iterations
eval_interval = 500
learning_rate = 1e-4 # Lower learning rate for stable training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🖥️  Using device: {device}")

# Encode dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(f"📊 Encoded data length: {len(data):,}")

# Train/validation split
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # Ensure we don't go out of bounds
    max_start = len(data_source) - block_size - 1
    if max_start <= 0:
        print("❌ Dataset too small for the block size!")
        return None, None
    
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x, y


""" Phase 3: Transformer Architecture Components """

class Head(nn.Module):
    """Single attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores with proper scaling
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Fixed: was self.head
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU works better than ReLU for transformers
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block with attention and feed-forward"""
    def __init__(self, n_embd, n_head):
        super().__init__()  # Fixed: was super.__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-layer norm (more stable training)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


""" Phase 4: Complete GPT Model """

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)    # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text with temperature and top-k sampling"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to last block_size tokens
                idx_cond = idx[:, -block_size:]
                
                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax and sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


""" Phase 5: Training Loop """

# Initialize model
model = GPTLanguageModel()
model = model.to(device)
print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20)  # More samples for better estimate
        for k in range(20):
            X, Y = get_batch(split)
            if X is None:  # Handle small dataset case
                continue
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("🏋️ Starting training...")
best_val_loss = float('inf')

for iter in range(max_iters):
    # Evaluate loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {iter:5d}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, LR {current_lr:.2e}")
        
        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': (stoi, itos),
                'config': {
                    'vocab_size': vocab_size,
                    'block_size': block_size,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'dropout': dropout
                }
            }, 'best_harry_potter_model.pth')

    # Training step
    xb, yb = get_batch('train')
    if xb is None:
        continue
        
    xb, yb = xb.to(device), yb.to(device)

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()

print("✅ Training completed!")


""" Phase 6: Save Model and Vocabulary """
# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab': (stoi, itos),
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'dropout': dropout
    }
}, 'harry_potter_model_final.pth')

print("💾 Model saved as 'harry_potter_model_final.pth'")


""" Phase 7: Text Generation and Interactive Chat """

def load_model(model_path):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Restore configuration
    config = checkpoint['config']
    global vocab_size, block_size, n_embd, n_head, n_layer, dropout
    vocab_size = config['vocab_size']
    block_size = config['block_size']
    n_embd = config['n_embd']
    n_head = config['n_head']
    n_layer = config['n_layer']
    dropout = config['dropout']
    
    # Restore vocabulary
    global stoi, itos
    stoi, itos = checkpoint['vocab']
    
    # Create and load model
    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def generate_text(model, prompt, max_tokens=150, temperature=0.8, top_k=50):
    """Generate text from prompt"""
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    generated_ids = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    generated_text = decode(generated_ids)
    
    # Return only the new generated part
    return generated_text[len(prompt):]

# Interactive chat mode
def interactive_chat():
    """Interactive chat with the Harry Potter model"""
    try:
        # Try to load the best model first
        if os.path.exists('best_harry_potter_model.pth'):
            model = load_model('best_harry_potter_model.pth')
            print("📚 Loaded best Harry Potter model!")
        else:
            model = load_model('harry_potter_model_final.pth')
            print("📚 Loaded final Harry Potter model!")
    except:
        print("❌ No trained model found! Please run training first.")
        return

    print("\nYo bruh!...sup bruv")
    print("✨ gimme me somes shit to continue!")
    print("💡 Type 'exit' to quit, 'temp X' to change temperature (0.1-2.0)")

    temperature = 0.8
    
    while True:
        try:
            user_input = input("\n🧙 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("🪄 Farewell, and may the magic be with you!")
                break
            
            # Temperature control
            if user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️ Temperature set to {temperature}")
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except:
                    print("❌ Invalid temperature format. Use: temp 0.8")
                continue
            
            if not user_input:
                continue
            
            print("Well I might say...")
            
            # Generate response
            response = generate_text(model, user_input, max_tokens=200, temperature=temperature, top_k=50)
            
            print(f"🪄 Sup! Texcore with ya: {user_input}{response}")
            
        except KeyboardInterrupt:
            print("\n🪄 Farewell, and may the magic be with you!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Start interactive chat if this script is run directly
if __name__ == "__main__":
    # If model exists, go straight to chat
    if os.path.exists('texcore.pth') or os.path.exists('texcore.pth'):
        interactive_chat()
    else:
        print("🏋️ Training will start automatically. After training, the chat will begin!")
        # Training code already ran above, now start chat
        interactive_chat()