import os

class DataStream(object):
    
    def __init__(
        self,
        batcher,
        transformer,
        formatter,
        transform_controls,
        batch_size,
    ):
        self.batcher = batcher
        self.transformer = transformer
        self.formatter = formatter
        self.transform_controls = transform_controls
        self.batch_size = batch_size
    
    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch_ids = self.batcher.get_ids(batch_size=batch_size)
        data = self.transformer(
            batch_ids, 
            controls=self.transform_controls
        )
        formatted = self.formatter(data)
        return formatted
    
    def __iter__(self):
        n_batches = self.batcher.iterations_per_epoch(self.batch_size)
        for i in range(n_batches):
            yield self.get_batch()
