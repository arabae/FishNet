import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fishnet import FishNet


class Trainer():
    def __init__(self, args, dataloaders):
        super(Trainer, self).__init__()
        if args.isTrain:
            self.train_loder, self.valid_loader = dataloaders[0], dataloaders[1]
        else:
            self.test_loader = dataloaders

        self.epoch = args.epoch
        self.device = args.device
        self.bs = args.batch_size

        self.best_acc = 0
        self.model = FishNet(args).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90], gamma=0.1)

        self.model_path = args.model_path

    def training(self):
        for e in range(1, self.epoch+1):
            total_loss, total_top1_acc, total_top5_acc = 0, 0, 0

            for batch_idx, (images, labels) in enumerate(self.train_loder):
                self.optimizer.zero_grad()

                images, labels = images.float().to(self.device), labels.to(self.device)
                self.model.train()
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                top1_acc, top5_acc = self.top_k_accuracy(preds.data, labels, topk=(1, 5))

                loss.backward()

                total_loss += loss.item()
                total_top1_acc += top1_acc
                total_top5_acc += top5_acc

                self.optimizer.step()

                if (batch_idx+1) % 200 == 0:
                    print(f'[{batch_idx+1:4d}/{len(self.train_loder)}] -- train loss: {total_loss/(batch_idx+1):.5f} || top1 acc: {total_top1_acc/(batch_idx+1):.5f} || top5 acc: {total_top5_acc/(batch_idx+1):.5f}')
                    '''
                    self.wandb.log({
                        'train_batch_loss': total_loss/(batch_idx+1),
                        'train_batch_top1': total_top1_acc/(batch_idx+1),
                        'train_batch_top5': total_top5_acc/(batch_idx+1)
                    })
                    '''
            
            print()
            print(f'**[{e:3d}/{self.epoch}] -- train epoch loss: {total_loss/len(self.train_loder):.5f} || top1 epoch acc: {total_top1_acc/len(self.train_loder):.5f} || top5 epoch acc: {total_top5_acc/len(self.train_loder):.5f}')
            '''
            self.wandb.log({
                'epoch': e,
                'train_loss': total_loss/len(self.train_loder),
                'train_top1': total_top1_acc/len(self.train_loder),
                'train_top5': total_top5_acc/len(self.train_loder),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            '''

            self.validation(e)
            self.scheduler.step()
        
    def validation(self, e):
        valid_loss, valid_top1_acc, valid_top5_acc = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(self.valid_loader):
            self.model.eval()
            with torch.no_grad():
                images, labels = images.float().to(self.device), labels.to(self.device)
                
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                top1_acc, top5_acc = self.top_k_accuracy(preds.data, labels, topk=(1, 5))

                valid_loss += loss.item()/len(self.valid_loader)
                valid_top1_acc += top1_acc/len(self.valid_loader)
                valid_top5_acc += top5_acc/len(self.valid_loader)
        
        print(f'**[{e:3d}/{self.epoch}] -- valid total loss: {valid_loss:.5f} || valid top1 acc: {valid_top1_acc:.5f} || valid top5 acc: {valid_top5_acc:.5f}')
        '''
        self.wandb.log({
                'epoch': e,
                'valid_loss': valid_loss,
                'valid_top1': valid_top1_acc,
                'valid_top5': valid_top5_acc
            })
        '''

        if self.best_acc < valid_top1_acc:
            print(f"New best model for validation top1 accuracy : {valid_top1_acc:.5f}! saving the best model..")
            torch.save(self.model.state_dict(), f"{self.model_path}/best.pt")
            self.best_acc = valid_top1_acc
        
        print("==================================================================")

    def inference(self):
        self.model.load_state_dict(torch.load(f"{self.model_path}/best.pt"))
        self.model.eval()
        
        test_total_loss, test_top1_acc, test_top5_acc = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.float().to(self.device), labels.to(self.device)
                
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                top1_acc, top5_acc = self.top_k_accuracy(preds.data, labels, topk=(1, 5))

                test_total_loss += loss.item()/len(self.test_loader)
                test_top1_acc += top1_acc/len(self.test_loader)
                test_top5_acc += top5_acc/len(self.test_loader)

            print("============== Best Inference Results ==============")
            print(f"Top-1 Accuracy : {test_top1_acc:.5f}")
            print(f"Top-5 Accuracy : {test_top5_acc:.5f}")
    
    def top_k_accuracy(self, output, target, topk=(1, )):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / self.bs)[0])
        return res
