# Import PyTorch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F

import torchvision.utils as vutils


# Import neural net
from neural_net import Generator, Discriminator

# Import utulities
# from utils import *
import utils
from utils import *

# Import other useful stuff
import logging
import sagemaker_containers
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

env = sagemaker_containers.training_env()

# Setup the summaries writer
from tensorboardX import SummaryWriter
tensorboard_logdir = os.path.join(env.output_data_dir, 'runs')
writer = SummaryWriter(tensorboard_logdir)


def load_model(args):
    """
    Load and connfigure the Generator, and the Discriminator networks

    Returns:
    Generator, Discriminator
    """
    
    # Create and initialize the Generator
    Gen = Generator(z_size = args.z_size,
                    out_size=args.channel_size,
                    ngf = args.ngf)

    print("Generator network: ")
    print(Gen)

    # Create and initialize the Discriminator
    Disc = Discriminator(in_size=args.channel_size, ndf=args.ndf)
    print("Discriminator network: ")
    print(Disc)

    return Gen, Disc


def _train(args):
    """
    Training script. 6 steps:

    1. Check if there are more than 1 hosts, if yes, start a distributed training.
        The backend can be selected with the argument --dist-backend. Default is gloo.
    
    2. Check if there are GPUs available. If yes, set the 'device' to GPU.

    3. Load the data with a training loader.

    4. Load the model and move it to the 'device' (CPU or GPU)

    5. Set the Criterion (loss function) and the optimizers

    6. Start the training loop
    """

    # 1. Distributed training

    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        print(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    # 2. Set the device to Cuda if a GPU is found, else use CPU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    
    # 3. Load the dataset

    print("Loading the Celeb-A dataset")
    
    train_loader = get_train_data_loader(args)
    
    # 4.a Load the model

    print("Loading the model")
    Gen, Disc = load_model(args)
    
    print("Model loaded")
    

    # 4.b move the model to the right device

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        Gen = nn.DataParallel(Gen)
        Disc = nn.DataParallel(Disc)

    Gen = Gen.to(device)
    Disc = Disc.to(device)

    
    # 5. Set the loss function and optimizers

    # 5.1 Loss function
    criterion = nn.BCELoss().to(device)

    # 5.2 Optimizers; one for each network
    optimizer_Disc = torch.optim.Adam(Disc.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_Gen = torch.optim.Adam(Gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 5.3 Setup additional variables to track metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Disc_losses = AverageMeter()
    Gen_losses = AverageMeter()

    with torch.set_grad_enabled(False):
        fixed_noise = torch.FloatTensor(8 * 8, args.z_size, 1, 1).normal_(0, 1)
        fixed_noise = fixed_noise.to(device)

    # fixed_noise = Variable(fixed_noise.cuda(), volatile=True)

    
    # 6. Start the training loop

    end = time.time()

    Disc.train()
    Gen.train()

    Disc_loss_list = []
    Gen_loss_list = []

    print("Starting training loop.")
 
    for epoch in range(0, args.epochs):
        for i, (real_faces,_) in enumerate(train_loader):
            
            """
            ****Discriminator training update****

            The Discriminator maximizes [log(D(x)) + log(1 - D(G(z)))]
            """

            data_time.update(time.time() - end)

            real_faces = real_faces.to(device)

            batch_size = real_faces.size(0)

            # Training Discriminator on real data: D(x)

            # Convention: Use real labels as 1, and fake labels as 0

            real_label = torch.ones(batch_size).to(device)

            Disc_real_output = Disc(real_faces).squeeze()

            """
            Loss function for the Discriminator to learn what are real images.
            Remember:
               - Real label = 1
               - Fake label = 0

            L = -w[1 * log(D(x)) + 0 * log(1 - D(x))] = -log(D(x)) if we consider the weights to be 1.
            Minimizing -log(D(x)) ==> Maximizing log(D(x))

            """

            Disc_real_loss = criterion(Disc_real_output, real_label)

            # Training Discriminator on fake data

            fake_label = torch.zeros(batch_size).to(device)

            # The randomly initialized z vector
            noise = torch.randn((batch_size, args.z_size)).view(-1, args.z_size, 1, 1).to(device)

            # Generate the fake images

            Gen_output = Gen(noise)

            # Pass the Generator's fake images to the Discriminator

            Disc_fake_output = Disc(Gen_output.detach()).squeeze()

            """
            Discriminator fake loss:
            L = -w[0 * log(D(G(x))) + 1 * log(1 - D(G(z)))]
            Consider w = 1
            => L = -log(1 - D(G(x)))
            ==> Minimizing L is equivalent to Maximixing log(1 - D(G(z)))

            """
            Disc_fake_loss = criterion(Disc_fake_output, fake_label)

            """
            Total Discriminator loss:
            L = - [log(D(x)) + log(1 - D(G(z)))]
            ==> minimizing L is eq to maximixing [log(D(x)) + log(1 - D(G(x)))]
            """

            Disc_training_loss = Disc_real_loss + Disc_fake_loss

            # Update the average meter for Discriminator losses

            Disc_losses.update(Disc_training_loss.item()) # .item() PyTorch 1.0
            
            
            # Discriminator training

            Disc.zero_grad()
            Disc_training_loss.backward()
            optimizer_Disc.step()


            """
            ****Generator training update****

            The Generator maximizes log(D(G(z)))

            Remember: FAKE labels are REAL for the Generator loss function.

            L = -w[1 * log(D(G(z))) + 0 * log(1 - D(G(z)))]

            Assuming the weight is 1

            => L = -log(D(G(z)))

            i.e Minimizing L is equivalent to Maximizing log(DG(z))
            """

            Gen.zero_grad()
            gen_label = real_label.new_ones(batch_size)
            # Gen_output is created when training the Discriminator

            output = Disc(Gen_output).squeeze()

            Gen_training_loss = criterion(output, gen_label)

            # Update the average meter for Generator losses

            Gen_losses.update(Gen_training_loss.item())

            Gen_training_loss.backward()

            optimizer_Gen.step()

            ### Update the batch time

            batch_time.update(time.time() - end)
            end = time.time()
            
            ### Logging Summaries on Tensorboard

            n_iter = epoch * len(train_loader) + i
            D_x = Disc_real_output.data.mean()
            D_G_z1 = Disc_fake_output.data.mean()
            D_G_z2 = output.data.mean()

            # Average batch time
            writer.add_scalar('AverageBatchTime', batch_time.avg, n_iter)

            # Average data loading time
            writer.add_scalar('AverageDataTime', data_time.avg, n_iter)

            
            # Average discriminator loss
            writer.add_scalar('AverageLoss/Discriminator', Disc_losses.avg, n_iter)
            
            # Average Generator loss
            writer.add_scalar('AverageLoss/Generator', Gen_losses.avg, n_iter)
            
            # Discriminator loss
            writer.add_scalar('Loss/Discriminator', Disc_training_loss.item(), n_iter)
            
            # Generator loss
            writer.add_scalar('Loss/Generator', Gen_training_loss.item(), n_iter)
            
            # Batch-average Discriminator predictions. 
            #    For real images
            writer.add_scalar('D(x)', D_x, n_iter)
            
            #    For the fakes used to train the Discriminator
            writer.add_scalar('D(G(z1))', D_G_z1, n_iter)
            
            #    For the fakes used to train the Generator
            writer.add_scalar('D(G(z2))', D_G_z2, n_iter)


            if i % args.display_after == 0:
                print_log(epoch + 1, args.epochs, i + 1, len(train_loader), args.lr,
                          args.display_after, batch_time, data_time, Disc_losses, Gen_losses)
                batch_time.reset()
                data_time.reset()

                faces = real_faces.to('cpu')
                fake_faces = Gen_output.to('cpu')

                vutils.save_image(faces,
                    '%s/real_samples.png' % env.output_data_dir,
                    normalize=True)
                writer.add_image('real_samples', vutils.make_grid(faces, normalize=True), n_iter)
            
                vutils.save_image(fake_faces,
                    '%s/fake_samples_epoch_%03d.png' % (env.output_data_dir, epoch),
                    normalize=True)
                    
                writer.add_image('fake_samples', vutils.make_grid(fake_faces, normalize=True), n_iter)

            elif (i + 1) == len(train_loader):
                print_log(epoch + 1, args.epochs, i + 1, len(train_loader), args.lr,
                          (i + 1) % args.display_after, batch_time, data_time, Disc_losses, Gen_losses)
                batch_time.reset()
                data_time.reset()


        ### Update Avg meters after every epoch

        Disc_loss_list.append(Disc_losses.avg)
        Gen_loss_list.append(Gen_losses.avg)
        Disc_losses.reset()
        Gen_losses.reset()

        # Plot the generated images and loss curve
        plot_result(Gen, fixed_noise, args.image_size, epoch + 1, env.output_data_dir, is_gray=False)

        plot_loss(Disc_loss_list, Gen_loss_list, epoch + 1, args.epochs, env.output_data_dir)


    create_gif(args.epochs, env.output_data_dir)
    print('Finished Training')


    return _save_model(Gen, Disc, args.model_dir)


def _save_model(gen, disc, model_dir):

    logger.info("Saving the models.")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(gen.cpu().state_dict(), os.path.join(model_dir, 'Generator.pth'))
    torch.save(disc.cpu().state_dict(), os.path.join(model_dir, 'Discriminator.pth'))


if __name__ == '__main__':
    
    args = parse()
    _train(args)
