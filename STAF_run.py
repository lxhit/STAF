import torch
import numpy as np
import argparse
import os
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy
from net_3d import Model
import tensorflow as tf
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import video_reader
from utils_n.torchtools import one_hot
from utils_n.losses import CrossEntropyLoss
def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.writer = SummaryWriter()
        
        gpu_device = 'cuda:%s' % self.args.start_gpu
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        self.criterion = CrossEntropyLoss(self.device)
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = Model(self.args)
        model = model.to(self.device)
        print(model)

        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2-cmn","ssv2-otam", "kinetics", "hmdb", "ucf"], default="ssv2", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=1, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=5000)
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=16, help="Frames per clip.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--data_folder", default=None, help="Directory to load the splits and dataset.")
        parser.add_argument("--start_gpu", type=int, default=0, help="starting number of gpu device")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--split", type=int, default=3, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--pretrained_3dmodels", default=False, action="store_true", help="Load pretrained 3d models")
        parser.add_argument("--pretrained_path", default=None, help="Directory of pretrained 3d models.")
        parser.add_argument("--scale_cls", type=int, default=7, help="scale_cls")
        parser.add_argument("--train_num_classes", type=int, default=64, help="total number of classes for global video classifier")
        parser.add_argument("--lamda", type=float, default=0.5, help="rate of two loss functions")
        parser.add_argument("--alpha", type=float, default=1, help="rate of two similar scores")
        parser.add_argument("--pre_data", choices=["KM", "k", "KMS","MS","M"], default="K", help="PRE-DATA")
        args = parser.parse_args()

        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if args.dataset == "ssv2-cmn":
            args.traintestlist = os.path.join(args.data_folder, "video_datasets/splits/ssv2_CMN")
            args.path = os.path.join(args.data_folder, "video_datasets/SSV2-CMN/video_frame")
        elif args.dataset == "ssv2-otam":
            args.traintestlist = os.path.join(args.data_folder, "video_datasets/splits/ssv2_OTAM")
            args.path = os.path.join(args.data_folder, "video_datasets/SSV2-OTAM/video_frame")
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.data_folder, "video_datasets/splits/kineticsTrainTestlist")
            args.path = os.path.join(args.data_folder, "video_datasets/Kinetics-100/video_frame")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.data_folder, "video_datasets/splits/ucf_ARN")
            args.path =os.path.join(args.data_folder, "video_datasets/UCF/video_frame")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.data_folder, "video_datasets/splits/hmdb_ARN")
            args.path = os.path.join(args.data_folder, "video_datasets/HMDB51/video_frame")

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.visible_device_list = '%s' % self.args.start_gpu
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations
                accuracy_dict_iteration = {}
                iteration = self.start_iteration
                for task_dict in self.video_loader:
                    if iteration >= total_iterations:
                        break
                    iteration += 1
                    torch.set_grad_enabled(True)
                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)
                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.scheduler.step()
                    if (iteration + 1) % self.args.print_freq == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item())
                                      )
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        self.save_checkpoint(iteration + 1)


                    if ((iteration + 1) % self.args.test_iters == 0) and (iteration + 1) != total_iterations:
                        accuracy_dict = self.test(session)
                        iter_str = str(iteration+1)
                        accuracy_dict_iteration[iter_str] = accuracy_dict
                        self.test_accuracies.print(self.logfile, accuracy_dict)
                        print(accuracy_dict_iteration)
                        print(self.max_accuracy_from_dict(accuracy_dict_iteration))

                # save the final model
                print(accuracy_dict_iteration)
                print(self.max_accuracy_from_dict(accuracy_dict_iteration))
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def max_accuracy_from_dict(self,accuracy_dict):
        max_dict = {}
        max_accuracy = 0.0
        max_confidence = 0.0
        best_iteration = 6400
        for key,value in accuracy_dict.items():
            accuracy_temp = value[self.args.dataset]['accuracy']
            if accuracy_temp > max_accuracy:
                max_accuracy = accuracy_temp
                max_confidence = value[self.args.dataset]['confidence']
                best_iteration = key
        max_dict['best_accuracy'] = max_accuracy
        max_dict['confidence'] = max_confidence
        max_dict['best_iteration'] = best_iteration
        return max_dict



    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels,real_target_labels = self.prepare_task(task_dict)
        context_train_1hot = one_hot(context_labels,self.args.way).cuda(self.device)
        target_labels_1hot = one_hot(target_labels,self.args.way).cuda(self.device)
        real_target_labels = real_target_labels.cuda(self.device)
        context_train_1hot = context_train_1hot.unsqueeze(0)
        target_labels_1hot = target_labels_1hot.unsqueeze(0)
        # convert to float
        context_train_1hot = context_train_1hot.float()
        target_labels_1hot = target_labels_1hot.float()


        ytest, cls_scores ,accuracy_score= self.model(context_images, target_images, context_train_1hot, target_labels_1hot)
        real_target_labels = real_target_labels.long()
        loss1 = self.criterion(ytest, real_target_labels.view(-1))
        loss2 = self.criterion(cls_scores, target_labels.view(-1))
        loss = loss1 + self.args.lamda * loss2
        loss = loss/self.args.tasks_per_batch
        loss.backward(retain_graph=False)

        batch_size = 1
        num_test_examples = self.args.query_per_class
        accuracy_score = accuracy_score.view(batch_size * num_test_examples, -1)
        labels_test = target_labels.view(batch_size * num_test_examples)

        _, preds = torch.max(accuracy_score.detach().cpu(), 1)
        acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
        return loss, acc

    def test(self, session):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracy_dict ={}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            for task_dict in self.video_loader:
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                context_images, target_images, context_labels, target_labels,real_target_labels = self.prepare_task(task_dict)
                context_train_1hot = one_hot(context_labels,self.args.way).cuda(self.device)
                target_labels_1hot = one_hot(target_labels,self.args.way).cuda(self.device)
                real_target_labels = real_target_labels.cuda(self.device)
                context_train_1hot = context_train_1hot.unsqueeze(0)
                target_labels_1hot = target_labels_1hot.unsqueeze(0)
                context_train_1hot = context_train_1hot.float()
                target_labels_1hot = target_labels_1hot.float()
                mytest, cls_scores ,accuracy_score= self.model(context_images, target_images, context_train_1hot, target_labels_1hot)
                batch_size = 1
                num_test_examples = self.args.query_per_class_test
                accuracy_score = accuracy_score.view(batch_size * num_test_examples, -1)
                labels_test = target_labels.view(batch_size * num_test_examples)

                _, preds = torch.max(accuracy_score.detach().cpu(), 1)
                acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
                accuracies.append(acc.item())
                if (iteration + 1) % self.args.print_freq == 0:
                    accuracy_print = np.array(accuracies).mean() * 100.0
                    confidence_print = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                    accuracy_dict[item] = {"accuracy": accuracy_print, "confidence": confidence_print}
                    print(accuracy_dict)

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
            self.video_loader.dataset.train = True
        self.model.train()
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)
        return context_images, target_images, context_labels, target_labels,real_target_labels


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
