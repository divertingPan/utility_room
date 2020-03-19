from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", default=96)
parser.add_argument("--ndf", default=96)
parser.add_argument("--nz", default=256)
parser.add_argument("--img_size", default=96)
parser.add_argument("--batch_size", default=25)
parser.add_argument("--lr1", default=0.0002)  # G的学习率
parser.add_argument("--lr2", default=0.0002)  # D的学习率
parser.add_argument("--beta1", default=0.5)
parser.add_argument("--epochs", default=36)
parser.add_argument("--save_every", default=100)
opt = parser.parse_args()

cudnn.benchmark = True


class NetD(nn.Module):
    # 构建一个判别器，相当与一个二分类问题, 生成一个值
    def __init__(self):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 输入96*96*3
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 输入32*32*ndf
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 输入16*16*ndf*2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # 输入为8*8*ndf*4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 输入为4*4*ndf*8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()  # 分类问题
        )

    def forward(self, x):
        return self.main(x).view(-1)


class NetG(nn.Module):
    # 定义一个生成模型，通过输入噪声来产生一张图片
    def __init__(self):
        super(NetG, self).__init__()
        ngf = opt.ngf
        self.main = nn.Sequential(
            # 假定输入为一张1*1*opt.nz维的数据(opt.nz维的向量)
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # 输入一个4*4*ngf*8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入一个8*8*ngf*4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入一个16*16*ngf*2
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 输入一个32*32*ngf
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # 输出一张96*96*3
        )

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = datasets.ImageFolder(r"img/train/cat",
                                   transform=transforms.Compose([transforms.RandomResizedCrop(opt.img_size,
                                                                                              scale=(0.9, 1.0),
                                                                                              ratio=(0.9, 1.1)),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor()]))
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            drop_last=True)
    # 初始化网络
    netg, netd = NetG().to(device), NetD().to(device)
    # 设定优化器参数
    optimize_g = torch.optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
    # 设置动态学习率
    scheduler_g = lr_scheduler.StepLR(optimize_g, step_size=10, gamma=0.9)
    scheduler_d = lr_scheduler.StepLR(optimize_d, step_size=10, gamma=0.9)

    loss_func = nn.BCELoss()
    # 定义标签, 并且开始注入生成器的输入noise
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    # 训练网络
    netg.train()
    netd.train()
    for epoch in range(opt.epochs):
        scheduler_d.step()
        scheduler_g.step()
        for i, img in enumerate(dataloader):
            real_img = img[0].to(device)  # dataloader里的img是个列表，第一列是图片第二列是类别
            # 训练判别器
            netd.zero_grad()
            # 真图
            real_out = netd(real_img * 2 - 1)
            error_d_real = loss_func(real_out, true_labels)
            error_d_real.backward()
            # 随机生成的假图
            noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
            fake_image = netg(noises)
            fake_out = netd(fake_image.detach())
            error_d_fake = loss_func(fake_out, fake_labels)
            error_d_fake.backward()
            optimize_d.step()

            # 计算loss
            error_d = error_d_fake + error_d_real
            if i % 20 == 0:
                print("\033[0m第{0}轮:\t判别网络 损失:{1:.5} 对真图评分:{2:.5} 对生成图评分:{3:.5} lr:{4}".format(epoch + 1,
                                                                                               error_d.item(),
                                                                                               real_out.data.mean(),
                                                                                               fake_out.data.mean(),
                                                                                               optimize_g.param_groups[0]['lr']))
            # 训练生成器
            netg.zero_grad()
            # noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            # fake_image = netg(noises)
            output = netd(fake_image)
            error_g = loss_func(output, true_labels)
            error_g.backward()
            optimize_g.step()

            if i % 20 == 0:
                print("\t\t\033[32m生成网络 损失:{0:.5}".format(error_g.item()))
            # 保存模型和图片
            if i % opt.save_every == 0:
                fix_fake_image = netg(fix_noises) * 0.5 + 0.5
                # save_image(real_img, "img/generate/cat/{0}-{1}-real_img.jpg".format(epoch, i), nrow=5)
                save_image(fix_fake_image.detach(), "img/generate/cat/{0}-{1}-fake_img.jpg".format(epoch, i), nrow=5)
                torch.save(netd, "model/cat/netd.pth")
                torch.save(netg, "model/cat/netg.pth")
                print("\033[34m图片与模型已保存")
