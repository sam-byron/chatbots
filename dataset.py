import math
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, tokenized_texts, block_size=256):
        """
        Initialize the dataset with tokenized texts and block size.
        Precompute all chunks for index-based access.
        """
        self.tokenized_texts = tokenized_texts
        self.block_size = block_size
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        """
        Precompute all chunks from the tokenized texts.
        Each chunk is of size `block_size`.
        """
        chunks = []
        for ids in self.tokenized_texts:
            for i in range(0, len(ids), self.block_size):
                chunk = ids[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    chunks.append(chunk)
        return chunks

    def __len__(self):
        """
        Return the total number of chunks.
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Return the chunk at the specified index.
        """
        return self.chunks[idx]