import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
from PIL import Image
import imageio
import importlib
import pip
import subprocess
import argparse


import logging
import sagemaker_containers

env = sagemaker_containers.training_env()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Required to install additional libraries like tensorflow, tensorboard, etc..
def _install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        logger.warning('Could not find required package {}, attempting to install...'.format(package))
        subprocess.call([sys.executable, '-m', 'pip', 'install', package])
    finally:
        globals()[package] = importlib.import_module(package)
        logger.info('Successfully installed and imported {}'.format(package))
    

def install_requirements():
    with open(os.path.join(env.model_dir,'requirements.txt'), 'r') as f:
        packages = f.read()
    
    packages = packages.split()[:-1]
    
    for package in packages:
        _install_and_import(package)
        
#     
class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_transform():
    image_size = 64
    return transforms.Compose([transforms.Scale(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])

def get_train_data_loader(args):
    transform = _get_transform()

    train_loader = torch.utils.data.DataLoader(
             datasets.ImageFolder(args.data_dir,
                transform=transform),
                batch_size=args.batch_size, shuffle=True,
              num_workers=args.workers, pin_memory=True)
    
    return train_loader



# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def read_data_file(filename, root=None):
    
    lists = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            if root is not None:
                info[0] = os.path.join(root, info[0])
            if len(info) == 1:
                item = (info[0], 1)
            else:
                item = (info[0], int(info[1]))
            lists.append(item)
            line = fp.readline()
    
    return lists

def pil_loader(path):
    
    img = Image.open(path)
    return img.convert('RGB')



def plot_loss(d_loss, g_loss, epoch, epochs, save_dir):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0,epochs + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(epoch)))
    plt.close()

def plot_result(G, fixed_noise, image_size, epoch, save_dir, fig_size=(8, 8), is_gray=False):

    G.eval()
    generated_images = G(fixed_noise)
    G.train()
    
    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    
    for ax, img in zip(axes.flatten(), generated_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, title, ha='center')
    
    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(epoch)))
    plt.close()

def create_gif(epochs, save_dir):
    
    images = []
    for i in range(1, epochs + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result.gif'), images, fps=5)
    
    images = []
    for i in range(1, epochs + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result_loss.gif'), images, fps=5)

def save_checkpoint(state, filename='checkpoint'):
    
    torch.save(state, filename + '.pth.tar')


def print_log(epoch, epochs, iteration, iters, learning_rate,
              display, batch_time, data_time, Disc_losses, Gen_losses):
    print('epoch: [{}/{}] iteration: [{}/{}]\t'
          'Learning rate: {}'.format(epoch, epochs, iteration, iters, learning_rate))

    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
          'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
          'Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'
          'Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'.format(
              display, batch_time=batch_time,
              data_time=data_time, loss_D=Disc_losses, loss_G=Gen_losses))
    print(time.strftime('%Y-%m-%d %H:%M:%S --------------------------------------------------------------------------------------------------\n', time.localtime()))


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=6, metavar='W',
                        help='number of data loading workers (default: 6)')

    parser.add_argument('--epochs', type=int, default=25, metavar='E',
                        help='number of total epochs to run (default: 25)')

    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='initial learning rate (default: 0.002)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--z_size', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--channel_size', type=int, default=3, help='The number of channels in the images')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--manualSeed', type=int, help='manual seed')


    
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    return parser.parse_args()

