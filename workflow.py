from torch import nn, optim, no_grad, save, load
from torch.utils import data
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

def train(model: nn.Module, optimiser: optim.Optimizer, data: data.DataLoader, metrics: Dict[str, nn.Module], epoch, device):
    with tqdm(data) as batches:
        for inputs, labels in batches:
            batches.set_description(f'Training: Epoch {epoch}')

            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()

            outputs = model(inputs)

            loss = metrics['loss'](outputs, labels)
            loss.backward()

            optimiser.step()

            results = {}
            for name, metric in metrics.items():
                results[name] = metric(outputs, labels).item()

            batches.set_postfix(results)

    return results

def validate(model: nn.Module, data: data.DataLoader, metrics: Dict[str, nn.Module], epoch, device, name: str = "Validation"):
    with no_grad():
        with tqdm(data) as batches:
            for inputs, labels in batches:
                batches.set_description(f'{name}: Epoch {epoch}')

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                results = {}
                for name, metric in metrics.items():
                    results[name] = metric(outputs, labels).item()

                batches.set_postfix(results)

    return results
    
def test(model: nn.Module, data: data.DataLoader, metrics: Dict[str, nn.Module], device):
    return validate(model, data, metrics, 0, device, name="Testing")

class ModelTrainer():

    def __init__(self, model: nn.Module, optimiser: optim.Optimizer, data: Tuple[data.DataLoader, data.DataLoader, data.DataLoader], metrics: Dict[str, nn.Module], epochs: int, device, save_dir: Path = None, checkpoint = False, continue_training = False):
        self.model = model
        self.optimiser = optimiser

        self.traindata = data[0]
        self.validdata = data[1]
        self.testdata = data[2]

        self.train_metrics = metrics['train']
        self.train_metrics.update(metrics['all'])

        self.valid_metrics = metrics['valid']
        self.valid_metrics.update(metrics['all'])

        self.test_metrics = metrics['test']
        self.test_metrics.update(metrics['all'])

        self.device = device

        self.save_dir = save_dir
        if checkpoint:
            self.checkpoint = save_dir / 'checkpoint.pt'

        self.epochs = epochs

        self.continue_training = continue_training

    def start(self):
        start_epoch = 0
        if self.continue_training:
            checkpoint_data = load(self.checkpoint)
            start_epoch = checkpoint_data['epoch']
            self.model.load_state_dict(checkpoint_data['model'])
            self.optimiser.load_state_dict(checkpoint_data['optim'])

        for epoch in range(start_epoch, self.epochs):
            train(self.model, self.optimiser, self.traindata, self.train_metrics, epoch, self.device)
            validate(self.model, self.validdata, self.valid_metrics, epoch, self.device)

            if self.checkpoint != False:
                save({
                    'model':self.model.state_dict(),
                    'optim':self.optimiser.state_dict(),
                    'epoch':epoch
                    }, self.checkpoint)          

        test(self.model, self.testdata, self.test_metrics, self.device)
        save({
            'model':self.model.state_dict()
        }, self.save_dir / 'final_model.pt')

# next: handle data parallel models for loading and saving. Data loading for dataframe style format

