import math
from torch.utils.data import IterableDataset, get_worker_info

class ChatDataset(IterableDataset):
    def __init__(self, tokenized_texts, block_size=256):
        self.tokenized_texts = tokenized_texts
        self.block_size = block_size
        self.total_chunks = self._calculate_total_chunks()

    def _calculate_total_chunks(self):
        total_chunks = 0
        for ids in self.tokenized_texts:
            total_chunks += len(ids) // self.block_size
        return total_chunks

    def __len__(self):
        return self.total_chunks

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.tokenized_texts)
        else:
            per_worker = math.ceil(len(self.tokenized_texts) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.tokenized_texts))

        for ids in self.tokenized_texts[start:end]:
            for i in range(0, len(ids), self.block_size):
                chunk = ids[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    yield chunk