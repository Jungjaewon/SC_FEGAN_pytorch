import time
import datetime
import os
import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
import wandb
import random
import numpy as np
import torchvision.models as models
import glob

torch.backends.cudnn.benchmark = True

from model import Generator as G
from model import Discriminator as D
from torchvision.utils import save_image
from data_loader import get_loader


vgg_activation = dict()


def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output #.detach()

    return hook


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.wandb = config['TRAINING_CONFIG']['WANDB'] == 'True'
        self.seed = config['TRAINING_CONFIG']['SEED']

        if self.seed != 0:
            print(f'set seed : {self.seed}')
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # https://hoya012.github.io/blog/reproducible_pytorch/
        else:
            print('do not set seed')

        self.train_loader = get_loader(config, 'train')
        self.test_loader = get_loader(config, 'test')
        self.img_size    = config['TRAINING_CONFIG']['IMG_SIZE']
        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.single_conv = config['TRAINING_CONFIG']['SINGLE_CONV'] == 'True'

        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])

        self.d_LR_S = config['TRAINING_CONFIG']['D_L_SLOP']
        self.g_LR_S = config['TRAINING_CONFIG']['G_L_SLOP']

        self.lambda_g_sn = config['TRAINING_CONFIG']['LAMBDA_G_SN']
        self.lambda_g_gt = config['TRAINING_CONFIG']['LAMBDA_G_GT']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']
        self.lambda_g_style = config['TRAINING_CONFIG']['LAMBDA_G_STYLE']
        self.lambda_g_tv = config['TRAINING_CONFIG']['LAMBDA_G_TV']
        self.alpha = config['TRAINING_CONFIG']['ALPHA']

        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gt = config['TRAINING_CONFIG']['LAMBDA_D_GT']
        self.lambda_d_gp    = config['TRAINING_CONFIG']['LAMBDA_GP']

        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        #self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        #assert self.gan_loss in ['lsgan', 'wgan', 'vanilla', 'r1loss']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        #if self.gan_loss == 'lsgan':
        #    self.adversarial_loss = torch.nn.MSELoss()
        #elif self.gan_loss == 'vanilla':
        #    self.adversarial_loss = torch.nn.BCELoss()

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')

        # vgg activation
        self.target_layer = ['pool_6', 'pool_13', 'pool_26']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = osp.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = osp.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])
        self.mode_arch_txt = osp.join(self.train_dir,'model_arch.txt')

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.test_step      = config['TRAINING_CONFIG']['TEST_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']

        self.lr_decay_policy = config['TRAINING_CONFIG']['LR_DECAY_POLICY']
        print(f'lr_decay_policy : {self.lr_decay_policy}')

        if self.wandb:
            wandb.login(key='3b3fd7ec86b8f3f0f32f2d7a78456686d8755d99')
            wandb.init(project='SC_FEGAN_pytorch', name=self.train_dir)

        self.build_model()

    def build_model(self):
        self.G, self.D = G(LR=self.g_LR_S, single_conv=self.single_conv).to(self.gpu), D(LR=self.d_LR_S).to(self.gpu)
        self.vgg = models.vgg19_bn(pretrained=True)

        for layer in self.target_layer:
            self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))
        self.vgg = self.vgg.eval().to(self.gpu)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        print(f'Use {self.lr_decay_policy} on training.')
        if self.lr_decay_policy == 'LambdaLR':
            self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)
            self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.5)
            self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.5)
        elif self.lr_decay_policy == 'StepLR':
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.8)
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.8)
        else:
            self.g_scheduler, self.d_scheduler = None, None

        if osp.exists(self.mode_arch_txt):
            os.remove(self.mode_arch_txt)

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(self.mode_arch_txt, 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)
            print('', file=fp)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def r1loss(self, inputs, label=None):
        # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return F.softplus(l*inputs).mean()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def cal_percep_style_loss(self, in_image, gt_image):

        fake_activation = dict()
        real_activation = dict()

        self.vgg(in_image)
        for layer in self.target_layer:
            fake_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        self.vgg(gt_image)
        for layer in self.target_layer:
            real_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        g_loss_percep, g_loss_style = 0, 0
        for layer in self.target_layer:
            g_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])
            fake_gram = self.gram_matrix(fake_activation[layer])
            gt_gram = self.gram_matrix(real_activation[layer])
            g_loss_style += self.l1_loss(fake_gram, gt_gram)

        fake_activation.clear()
        real_activation.clear()

        return g_loss_percep, g_loss_style

    def load_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*.ckpt'))

        if len(ckpt_list) == 0:
            return 0
        else:
            last_ckpt = sorted(ckpt_list,
                               key=lambda x: int(x.split(os.sep)[-1].replace('-model.ckpt', '')))[-1]

            ckpt_dict = torch.load(last_ckpt)
            print(f'ckpt_dict key : {ckpt_dict.keys()}')

            self.G.load_state_dict(ckpt_dict['G'])
            self.g_optimizer.load_state_dict(ckpt_dict['G_optim'])

            self.D.load_state_dict(ckpt_dict['D'])
            self.d_optimizer.load_state_dict(ckpt_dict['D_optim'])

            print(f'All models are load from {last_ckpt},')

            return int(osp.basename(last_ckpt).replace('-model.ckpt', ''))

    def train(self):

        # Set data loader.
        data_loader = self.train_loader
        iterations = len(self.train_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        fface_id, fface_in, fmask, fsketch, fcolor, fnoise, fface_gt = next(data_iter)
        fixed_data = fface_id, fface_in, fmask, fsketch, fcolor, fnoise, fface_gt
        fixed_data = [x.to(self.gpu) for x in fixed_data]
        fface_id, fface_in, fmask, fsketch, fcolor, fnoise, fface_gt = fixed_data
        fixed_data_cat = torch.cat([fface_in, fmask, fsketch, fcolor, fnoise], dim=1)

        epoch_r = self.load_model()

        start_time = time.time()
        print('Start training...')

        for e in range(epoch_r, self.epoch):
            for i in range(iterations):

                try:
                    _, face_in, mask, sketch, color, noise, face_gt = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, face_in, mask, sketch, color, noise, face_gt = next(data_iter)

                train_data = [face_in, mask, sketch, color, noise, face_gt]
                train_data = [x.to(self.gpu) for x in train_data]
                face_in, mask, sketch, color, noise, face_gt = train_data
                data_cat = torch.cat([face_in, mask, sketch, color, noise], dim=1)
                loss = dict()

                fake_image = self.G(data_cat)
                comp_image = fake_image * mask + face_gt * (1 - mask)

                if (i + 1) % self.d_critic == 0:
                    real_score = self.D(torch.cat([face_gt, mask, sketch, color, noise], dim=1))
                    fake_score = self.D(torch.cat([comp_image.detach(), mask, sketch, color, noise], dim=1))

                    real_loss = self.lambda_d_real * torch.mean(1 - real_score)
                    fake_loss = self.lambda_d_fake * torch.mean(1 + fake_score)
                    d_gt_loss = self.lambda_d_gt * torch.mean(torch.pow(real_score, 2))

                    d_loss = real_loss + fake_loss + d_gt_loss

                    alpha = torch.rand(face_gt.size(0), 1, 1, 1).to(self.gpu)
                    x_hat = (alpha * face_gt.data + (1 - alpha) * comp_image.data).requires_grad_(True)
                    out_src = self.D(torch.cat([x_hat, mask, sketch, color, noise], dim=1))
                    d_gp_loss = self.lambda_d_gp * self.gradient_penalty(out_src, x_hat)
                    d_loss += d_gp_loss

                    if torch.isnan(d_loss):
                        raise Exception('d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(e + 1, self.epoch, i + 1, iterations))

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss['D/real_loss'] = real_loss.item()
                    loss['D/fake_loss'] = fake_loss.item()
                    loss['D/gt_loss'] = d_gt_loss.item()
                    loss['D/gp_loss'] = d_gp_loss.item()
                    loss['D/d_loss'] = d_loss.item()

                if (i + 1) % self.g_critic == 0:

                    fake_score = self.D(torch.cat([comp_image, mask, sketch, color, noise], dim=1))

                    sn_loss = self.lambda_g_sn * -torch.mean(fake_score)

                    gt_loss = self.l1_loss(mask * fake_image, mask * face_gt) * self.alpha
                    gt_loss += self.l1_loss((1 - mask) * fake_image, (1 - mask) * face_gt) * self.lambda_g_gt

                    percep_loss1, style_loss1 = self.cal_percep_style_loss(fake_image, face_gt)
                    percep_loss2, style_loss2 = self.cal_percep_style_loss(comp_image, face_gt)

                    percep_loss = self.lambda_g_percep * (percep_loss1 + percep_loss2)
                    style_loss = self.lambda_g_style * (style_loss1 + style_loss2)

                    B, C, H, W = comp_image.size()
                    tv_loss = self.lambda_g_tv * (
                                torch.sum(torch.abs(comp_image[:, :, :, 1:] - comp_image[:, :, :, :-1])) +
                                torch.sum(torch.abs(comp_image[:, :, 1:, :] - comp_image[:, :, :-1, :])))
                    tv_loss /= (B * C * H * W)

                    g_loss = sn_loss + gt_loss + percep_loss + style_loss + tv_loss

                    if torch.isnan(g_loss):
                        raise Exception('d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(e + 1, self.epoch, i + 1, iterations))

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/sn_loss'] = sn_loss.item()
                    loss['G/tv_loss'] = tv_loss.item()
                    loss['G/percep_loss'] = percep_loss.item()
                    loss['G/style_loss'] = style_loss.item()
                    loss['G/gt_loss'] = gt_loss.item()
                    loss['G/g_loss'] = g_loss.item()

                if self.wandb:
                    for tag, value in loss.items():
                        wandb.log({tag: value})

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report = list()
                    fixed_fake = self.G(fixed_data_cat)

                    image_report.append(fixed_fake)
                    image_report.append(fixed_fake * fmask + fface_gt * (1 - fmask))
                    image_report.append(fface_gt)
                    image_report.append(fmask.expand_as(fface_gt))
                    image_report.append(fnoise.expand_as(fface_gt))
                    image_report.append(fcolor.expand_as(fface_gt))
                    image_report.append(fsketch.expand_as(fface_gt))
                    x_concat = torch.cat(image_report, dim=3)
                    # https://stackoverflow.com/questions/134934/display-number-with-leading-zeros/33860138
                    sample_path = osp.join(self.sample_dir, '{}-images.jpg'.format(str(e + 1).zfill(len(str(self.epoch)))))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # test step
            if (e + 1) % self.test_step == 0:
                self.test(self.test_loader, e + 1, 'test')

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:

                e_len = len(str(self.epoch))
                epoch_str = str(e + 1).zfill(e_len)
                ckpt_path = osp.join(self.model_dir, '{}-model.ckpt'.format(epoch_str))
                ckpt = dict()

                ckpt['G'] = self.G.state_dict()
                ckpt['D'] = self.D.state_dict()
                ckpt['G_optim'] = self.g_optimizer.state_dict()
                ckpt['D_optim'] = self.d_optimizer.state_dict()
                torch.save(ckpt, ckpt_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            if self.wandb:
                wandb.log({'G/lr': self.g_optimizer.param_groups[0]['lr']})
                wandb.log({'D/lr': self.d_optimizer.param_groups[0]['lr']})

            if self.lr_decay_policy != 'None':
                self.g_scheduler.step()
                self.d_scheduler.step()

        print('Training is finished')

    def test(self, data_loader, epoch, mode='test'):
        print(f'testing the model........')
        # Set data loader.
        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)
        fake_result_dir = osp.join(self.result_dir, f'{mode}_{epoch_str}')
        os.makedirs(fake_result_dir, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                f_id, face_in, mask, sketch, color, noise, face_gt = data
                test_data = [face_in, mask, sketch, color, noise, face_gt]
                test_data = [x.to(self.gpu) for x in test_data]
                face_in, mask, sketch, color, noise, face_gt = test_data
                test_cat = torch.cat([face_in, mask, sketch, color, noise], dim=1)
                image_report = list()
                fake_image = self.G(test_cat)
                comp_image = fake_image * mask + face_gt * (1 - mask)
                image_report.append(fake_image)
                image_report.append(comp_image)
                image_report.append(face_gt)
                image_report.append(mask.expand_as(face_gt))
                image_report.append(noise.expand_as(face_gt))
                image_report.append(color.expand_as(face_gt))
                image_report.append(sketch.expand_as(face_gt))
                x_report = torch.cat(image_report, dim=3)

                sample_path = osp.join(fake_result_dir, f'{str(f_id.item())}.jpg') # .zfill(filling)
                save_image(self.denorm(x_report.data.cpu()), sample_path, nrow=1, padding=0)