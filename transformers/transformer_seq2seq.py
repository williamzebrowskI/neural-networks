import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
import math
from tqdm import tqdm

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 16
EMBEDDING_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
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
        
        # Convert text to a list of token indices
        self.data = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
        self.data += [self.vocab['<eos>']] * MAX_SEQ_LENGTH  # Ensure the last sequence has enough length
        
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
    A Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, vocab_size):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size (int): The size of the vocabulary.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.pos_encoder = PositionalEncoding(EMBEDDING_SIZE)
        self.transformer = Transformer(d_model=EMBEDDING_SIZE, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                                       num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=FFN_HID_DIM)
        self.fc_out = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.
        
        Args:
            src (torch.Tensor): The source sequence.
            tgt (torch.Tensor): The target sequence.
        
        Returns:
            torch.Tensor: The output logits for the next token prediction.
        """
        # Apply embedding and positional encoding to the source and target sequences
        src = self.embedding(src) * math.sqrt(EMBEDDING_SIZE)
        tgt = self.embedding(tgt) * math.sqrt(EMBEDDING_SIZE)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Pass through the Transformer model
        output = self.transformer(src, tgt)
        
        # Apply the final linear layer to get the output logits
        return self.fc_out(output)

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

def main():
    """
    Main function to prepare the dataset, initialize the model, and train it.
    """
    # Prepare dataset
    dataset = TextDataset("book.txt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Calculate total number of batches
    total_sequences = len(dataset)
    total_batches = len(dataloader)
    
    # Print total sequences and batches
    print(f"Total Sequences: {total_sequences}")
    print(f"Total Batches per Epoch: {total_batches}")

    # Initialize model
    model = TransformerModel(len(dataset.vocab))
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, (src, tgt) in enumerate(dataloader):
            src = src.transpose(0, 1)
            tgt_input = tgt[:, :-1].transpose(0, 1)
            tgt_output = tgt[:, 1:].transpose(0, 1)
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, len(dataset.vocab)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:  # Adjust the printing frequency according to your dataset size
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}')
        print(f'End of Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}')

    # Save the model
    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    main()