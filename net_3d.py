import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet,cam_r3d




class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.scale_cls = args.scale_cls
        ## 2. init model
        self.base = resnet.R3D700(self.args.pretrained_path,pretrained=self.args.pretrained_3dmodels,pre_data=self.args.pre_data)
        self.cam = cam_r3d.CAM()
        self.nFeat = 2048
        # global classifier for
        self.classifier = nn.Conv3d(self.nFeat, self.args.train_num_classes, kernel_size=1)


    def test_cross(self, ftrain, ftest):
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(4)
        ftest = ftest.mean(5)
        ftest = ftest.mean(5)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def test_self(self, ftrain, ftest):
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(3)
        ftrain = ftrain.unsqueeze(1)
        ftest = ftest.mean(5)
        ftest = ftest.mean(4)
        ftest = ftest.mean(3)
        ftest = ftest.unsqueeze(2)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores
    def get_cls_score_cross(self,ftrain,ftest,batch_size,num_test):
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(4)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        ftrain_norm = ftrain_norm.unsqueeze(6)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        return cls_scores
    def get_cls_score_self(self,ftrain,ftest,batch_size,num_test):
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(3)
        ftrain = ftrain.unsqueeze(1)
        ftest_un = ftest.unsqueeze(2)
        ftest_norm = F.normalize(ftest_un, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        ftrain_norm = ftrain_norm.unsqueeze(6)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        return cls_scores
    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = 1, xtrain.size(0)
        num_test = xtest.size(0)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        x = torch.cat((xtrain, xtest), 0)
        label,f_x,f = self.base(x)
        _,_,d,h,w = f.size()
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain_self, ftest_self, ftrain_cross, ftest_cross = self.cam(ftrain, ftest)
        accuracy_score_self = self.test_self(ftrain_self, ftest_self)
        accuracy_score_cross = self.test_cross(ftrain_cross, ftest_cross)

        accuracy_score = torch.add(accuracy_score_cross, accuracy_score_self, alpha=self.args.alpha)
        cls_scores_self = self.get_cls_score_self(ftrain_self,ftest_self,batch_size,num_test)
        cls_scores_cross = self.get_cls_score_cross(ftrain_cross,ftest_cross,batch_size,num_test)

        cls_scores = torch.add(cls_scores_cross, cls_scores_self, alpha=self.args.alpha)


        ftest = ftest_cross
        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3) 
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, d,h, w)
        ytest = self.classifier(ftest)
        return ytest, cls_scores,accuracy_score




    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.base = torch.nn.DataParallel(self.base, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            self.cam = torch.nn.DataParallel(self.cam, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])



if __name__ == '__main__':

    class ArgsObject(object):
        def __init__(self):
            self.way = 5
            self.shot = 1
            self.query_per_class = 1
            self.seq_len = 8
            self.img_size = 84
            self.num_gpus = 1
            self.scale_cls=7
            self.alpha = 1
            self.pre_data = "KMS"
    args = ArgsObject()
    torch.manual_seed(0)
    net = Model(args)
    net.eval()
    x1 = torch.rand(15, 3, 16, 224, 224)
    x2 = torch.rand(2, 3, 16, 224, 224)
    y1 = torch.rand(1, 15, 5)
    y2 = torch.rand(1, 2, 5)

    ytest, cls_scores,accuracy_score = net(x1, x2, y1, y2)



