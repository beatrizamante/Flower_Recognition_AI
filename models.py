from os import path
import torch
from torch import nn, optim
from torchvision import models
from services import Classifier
from loader_controller import data_load
from utils import process_image

class model_load:
    
    criterion = nn.NLLLoss()

    def __init__(self, arch, epochs, lr, gpu_question, hidden_units, data_dir, save_dir):
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.hidden_units = hidden_units
        self.arch = arch
        self.epochs = epochs
        self.lr = lr
        self.gpu_question = gpu_question
        self.device = torch.device("cuda" if gpu_question else "cpu")
        self.checkpoint = {}
        self.model = self.__init_model()

    def __init_model(self):
        if self.arch == 'densenet121':
            model = models.densenet121(weights='DEFAULT')
        
            for param in model.parameters():
                param.requires_grad = False
            
            model.classifier = Classifier(self.hidden_units, 1024)
        else:
            model = models.vgg16(weights='DEFAULT')

            for param in model.parameters():
                param.requires_grad = False
            
            model.classifier = Classifier(self.hidden_units, 25088)

        model.to(self.device)

        return model

    def train(self):
        print("Training...")
        loaders = data_load(self.data_dir)
        optimizer = optim.Adam(self.model.classifier.parameters(), self.lr)
        steps = 0
        print_every = 5
        train_losses, valid_losses = [], []

        self.model.train()
        for self.epoch in range(self.epochs):
            running_loss = 0
            for images, labels in loaders.data_loaders['train']:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()

                    with torch.no_grad():
                        for images, labels in loaders.data_loaders['valid']:
                            images, labels = images.to(self.device), labels.to(self.device)
                            log_ps = self.model(images)
                            batch_loss = self.criterion(log_ps, labels)

                            valid_loss += batch_loss.item()

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(loaders.data_loaders['valid']))
                    valid_losses.append(valid_loss/len(loaders.data_loaders['valid']))

                    print(f"Epoch {self.epoch+1}/{self.epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(loaders.data_loaders['valid']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(loaders.data_loaders['valid']) * 100:.3f} %")
                    running_loss = 0
                    self.model.train()

        self.model.class_to_idx = loaders.data_loaders['train'].dataset.class_to_idx
        self.checkpoint = {
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'hidden_layers': [each.out_features for each in self.model.classifier.children() if isinstance(each, nn.Linear)],
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': self.model.class_to_idx
        }
        print("Completed.")

    def test(self, loader):
        print("Testing model...")
        accuracy = 0

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                log_ps = self.model(images)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(loader) * 100:.3f} %")

        print("Completed.")

    def save(self):
        filename = 'checkpoint.pth'
        f_path = path.join(self.save_dir, filename)

        torch.save(self.checkpoint, f_path)
        print("Checkpoint saved.")

    def predict(self, image_path, topk, gpu_question):
        image = process_image(image_path)
        image_input = torch.from_numpy(image).float()
        image_input = image_input.unsqueeze(0)
        self.device = torch.device("cuda" if gpu_question else "cpu")
        image_input = image_input.to(next(self.model.parameters()).device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_input)
            probabilities = torch.exp(output)

            top_p, top_class = probabilities.topk(topk, dim=1)

            top_p = top_p.squeeze().cpu().numpy()
            top_class = top_class.squeeze().cpu().numpy()

            idx_to_class = {value: key for key, value in self.model.class_to_idx.items()}
            top_classes = [idx_to_class[idx] for idx in top_class]

        print("Predicted probabilities.")
        return top_p, top_classes
    
