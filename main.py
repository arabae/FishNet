import torch
import pandas as pd

from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

from modules.args import get_args
from modules.trainer import Trainer
from modules.dataset import FishnetDataset
from modules.utils import ColorAugmentation, load_data, load_yaml, seed_everything, unpickle


if __name__=='__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize seed
    seed_everything(args.seed)
    
    whole_dataset = defaultdict(list)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262])

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ColorAugmentation(),
                                    normalize])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])
    if args.isTrain:
        print(">> Training Start!")
        print()
        
        # batch dataset -> whole dataset
        for idx in range(1, args.data_batch+1):
            load_data(f'{args.data_path}/data_batch_{idx}', whole_dataset, args.num_data_batch)
        whole_dataset = pd.DataFrame(whole_dataset)

        # split train/valid and set dataset
        train_, valid_ = train_test_split(whole_dataset, test_size=0.2)
        train_dataset, valid_dataset = FishnetDataset(train_, transform), FishnetDataset(valid_, test_transform)

        # set dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        # set trainer
        model_trainer = Trainer(args, (train_dataloader, valid_dataloader))

        # run train
        model_trainer.training()
    
    else:
        print(">> Inference Start!")
        print()

        load_data(f'{args.data_path}/test_batch', whole_dataset, args.num_data_batch)
        whole_dataset = pd.DataFrame(whole_dataset)

        # set dataloader
        test_dataset = FishnetDataset(whole_dataset, test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # set trainer
        model_trainer = Trainer(args, test_dataloader)
        
        # run inference
        model_trainer.inference()