import torch
import torch.nn as nn
import time

class RoboModelWrapper(nn.Module):
    def __init__(self, model, device='cuda', dtype=torch.float32):
        super().__init__()
        self.model = model.to(device, dtype)
    
    def unwrap(self):
        return self.model
    
    def _get_self_dtype_and_device(self):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return device, dtype

    def process_inputs(self, **inputs):
        device, dtype = self._get_self_dtype_and_device()
        def process_input(input_data):
            if isinstance(input_data, torch.Tensor):
                return input_data.to(device, dtype)
            elif isinstance(input_data, (list, tuple)):
                return type(input_data)(process_input(x) for x in input_data)
            elif isinstance(input_data, dict):
                return {k: process_input(v) for k, v in input_data.items()}
            else:
                return input_data
        return {k: process_input(v) for k, v in inputs.items()}
    
    def process_outputs(self, key_output, **outputs):
        def process_output(output_data):
            if isinstance(output_data, torch.Tensor):
                return output_data.detach().cpu()
            elif isinstance(output_data, (list, tuple)):
                return type(output_data)(process_output(x) for x in output_data)
            elif isinstance(output_data, dict):
                return {k: process_output(v) for k, v in output_data.items()}
            else:
                return output_data
        return process_output(key_output), {k: process_output(v) for k, v in outputs.items()}

    def forward(self, **inputs):
        self.train()
        processed_inputs = self.process_inputs(**inputs)
        loss, loss_dict = self.model(**processed_inputs)
        return loss, loss_dict
    
    def generate(self, **inputs):
        self.eval()
        processed_inputs = self.process_inputs(**inputs)
        with torch.no_grad():
            action, details = self.model.generate(**processed_inputs)
        action, details = self.process_outputs(action, **details)
        return action, details

class DataLoaderWithTimeWrapper:
    def __init__(self, dataloader, total_iters=200000):
        # set training steps
        self.total_iters = total_iters
        # set dataloader
        self.current_epoch = 0
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        
    def __iter__(self):
        current_iter = 0
        start_time = time.time()
        while current_iter < self.total_iters:
            batch = next(self.dataloader_iter)
            yield batch, time.time() - start_time
            start_time = time.time()
            current_iter += 1

    def __len__(self):
        return self.total_iters

class EPDataLoaderWithTimeWrapper:
    def __init__(self, dataloader, total_iters=200000):
        # set training steps
        self.total_iters = total_iters
        # set dataloader
        self.current_epoch = 0
        self.dataloader = dataloader
        self.dataloader.sampler.set_epoch(0)
        self.dataloader_iter = iter(self.dataloader)
        
    def __iter__(self):
        current_iter = 0
        start_time = time.time()
        while current_iter < self.total_iters:
            try:
                batch = next(self.dataloader_iter)
            except:
                self.current_epoch += 1
                print(f'switch to epoch {self.current_epoch}')
                self.dataloader.sampler.set_epoch(self.current_epoch)
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)
            
            yield batch, time.time() - start_time
            start_time = time.time()
            current_iter += 1

    def __len__(self):
        return self.total_iters
