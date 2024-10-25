import argparse
import json
from models import model_load
import torch
from torchvision import models
from services import Classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class for an input image using a trained deep learning model")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default: 5)')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        classifier = Classifier(
            checkpoint['hidden_units'],
            input_layer = 1024
        )
    else:
        model = models.vgg16(weights='DEFAULT')
        classifier = Classifier(
            checkpoint['hidden_units'],
            input_layer = 25088
        )

    placeholder = model_load(
        arch=checkpoint['arch'],
        epochs=1,
        lr=0000.1,
        gpu_question=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden_units=checkpoint['hidden_units'],
        data_dir="./",
        save_dir="./",
    )

  
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    placeholder.model = model

    return placeholder

def predict(model, image_path, top_k, use_gpu):
    return model.predict(image_path, top_k, use_gpu)

def load_category(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()


    
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(model, args.image_path, args.top_k, args.gpu)
    
    if args.category_names:
        cat_to_name = load_category(args.category_names)
        classes = [cat_to_name[str(clas)] for clas in classes]

    print("Probabilities - ", probs)
    print("Classes - ", classes)

if __name__ == '__main__':
    main()
