"""
This script implements a decoder-only Transformer model for text generation, similar to the architecture used in GPT (Generative Pre-trained Transformer). The model is trained on a text dataset, specifically the book "Pride and Prejudice," to learn to generate text in the style of the book.
The script trains the model on the text of "Pride and Prejudice" to generate text in a similar style. After training, the model can be used to generate text by predicting the next token in a sequence based on the previous tokens.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 32
EMBEDDING_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_DECODER_LAYERS = 3
MAX_SEQ_LENGTH = 128

def create_vocab(text):
    """
    Create a vocabulary from the input text.
    
    Args:
        text (str): The input text.
    
    Returns:
        dict: A dictionary mapping words to unique indices.
    """
    words = text.split()
    unique_words = set(words)
    vocab = {word: i+4 for i, word in enumerate(unique_words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab['<sos>'] = 2
    vocab['<eos>'] = 3
    return vocab

class TextDataset(Dataset):
    """
    A custom dataset class for loading text data.
    """
    def __init__(self, filepath, vocab=None):
        """
        Initialize the dataset.
        
        Args:
            filepath (str): Path to the text file.
            vocab (dict, optional): Predefined vocabulary. If None, a new vocabulary is created.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', ' ')

        if vocab is None:
            self.vocab = create_vocab(text)
        else:
            self.vocab = vocab
        
        self.data = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
        self.data += [self.vocab['<eos>']] * MAX_SEQ_LENGTH
        
    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.data) - MAX_SEQ_LENGTH + 1

    def __getitem__(self, idx):
        """
        Get a single sequence from the dataset.
        
        Args:
            idx (int): Index of the sequence.
        
        Returns:
            tuple: A tuple containing the input sequence and the target sequence.
        """
        sequence = self.data[idx:idx+MAX_SEQ_LENGTH]
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)
        return input_sequence, target_sequence

class TransformerModel(nn.Module):
    """
    A decoder-only Transformer model for text generation.
    """
    def __init__(self, vocab_size):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size (int): The size of the vocabulary.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.pos_encoder = PositionalEncoding(EMBEDDING_SIZE, MAX_SEQ_LENGTH)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBEDDING_SIZE, nhead=NHEAD, dim_feedforward=FFN_HID_DIM
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=NUM_DECODER_LAYERS
        )
        self.fc_out = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence to prevent attending to future tokens.
        
        Args:
            sz (int): The size of the mask.
        
        Returns:
            torch.Tensor: The generated mask.
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        """
        Forward pass of the Transformer model.
        
        Args:
            src (torch.Tensor): The input sequence.
        
        Returns:
            torch.Tensor: The output logits for the next token prediction.
        """
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        src = self.embedding(src) * math.sqrt(EMBEDDING_SIZE)
        src = self.pos_encoder(src)
        # Pass src as memory. In an autoregressive model, this is equivalent to the decoder attending to itself.
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=src_mask)
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding module to add positional information to the embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass to add positional encoding to the input embeddings.
        
        Args:
            x (torch.Tensor): The input embeddings.
        
        Returns:
            torch.Tensor: The embeddings with added positional encoding.
        """
        return x + self.encoding[:, :x.size(1)]

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): The model.
    
    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """
    Main function to prepare the dataset, initialize the model, and train it.
    """
    # Prepare dataset
    dataset = TextDataset("./book.txt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = TransformerModel(len(dataset.vocab))

    # Count parameters
    total_params = count_parameters(model)
    print(f"The model has {total_params:,} trainable parameters")
    
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=True)
        for i, (src, tgt) in progress_bar:
            src = src.transpose(0, 1)
            tgt_output = tgt.transpose(0, 1)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, len(dataset.vocab)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Color-coded logging
            if loss.item() < 1.0:
                color = Fore.GREEN
            elif loss.item() < 2.0:
                color = Fore.YELLOW
            else:
                color = Fore.RED
            
            progress_bar.set_postfix(loss=f"{color}{loss.item():.4f}{Style.RESET_ALL}")

        avg_loss = total_loss / len(dataloader)
        print(f'End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    main()