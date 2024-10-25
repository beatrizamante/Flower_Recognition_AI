import argparse
from models import model_load
from loader_controller import data_load

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Hello! In this program, you can train a new neural net by giving it a dataset of your choice.")
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture now: vgg16. Choose from: vgg16, densenet121')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs. Now: 4')
    parser.add_argument('--lr', type=float, default=0.0001, help='AI Learning Rate. Now: 0.0001')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Amount of Hidden Units. Now: 1024')
    parser.add_argument('--gpu', action='store_true', help='Boolean: Use of GPU to load application')
    parser.add_argument('--save_dir', type=str, default='./', help='Choose a checkpoint save directory')
    return parser.parse_args()

def create_model(args):
    return model_load(
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        lr=args.lr,
        hidden_units=args.hidden_units, 
        gpu_question=args.gpu,
        save_dir=args.save_dir
    )

def main():
    args = parse_args()
    model = create_model(args)
    loaders = data_load(args.data_dir)

    model.train()
    model.test(loaders.data_loaders['test'])
    model.save()

if __name__ == '__main__':
    main()
