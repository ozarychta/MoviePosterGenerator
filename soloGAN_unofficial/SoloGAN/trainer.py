"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import Generator,Discreminator
#from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
printout=False

class SoloGAN(nn.Module):
    def __init__(self, opt):
        super(SoloGAN, self).__init__()
        self.opt = opt
        self.nz = opt.style_dim # self.nz = 8

        self.lambda_cls = opt.lambda_cls
        self.lambda_rec = opt.lambda_rec
        self.lambda_cyc = opt.lambda_cyc
        self.lambda_latent = opt.lambda_latent

        # Initiate the networks
        self.gen = Generator(opt)
        self.dis = Discreminator(opt)

        # Setup the optimizers
        self.define_optimizers()
        # define adversarial loss
        self.adversarial_loss = torch.nn.MSELoss()

    def initialize(self):
        # Network weight initialization
        self.gen.apply(weights_init(self.opt.init_type))
        self.dis.apply(weights_init(self.opt.init_type))

    def define_optimizer(self, Net):
        beta1 = self.opt.beta1
        beta2 = self.opt.beta2
        return torch.optim.Adam(Net.parameters(),lr=self.opt.lr,betas=(beta1,beta2))

    def define_optimizers(self):
        """Define the optamizers for generator and discriminator."""
        self.gen_opt = self.define_optimizer(self.gen)
        self.dis_opt = self.define_optimizer(self.dis)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.gen_opt.zero_grad()
        self.gen_opt.zero_grad()

    def update_lr(self, lr):
        for param_group in self.gen_opt.param_groups:
            param_group['lr'] = lr
        for param_group in self.dis_opt.param_groups:
            param_group['lr'] = lr

    def recon_criterion(self, input, target):
         return torch.mean(torch.abs(input - target))

    def classification_loss(self, logit, target):
        return F.cross_entropy(logit, target) # """Compute binary or softmax cross entropy loss."""

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def get_z_random(self, batchSize, nz): #, random_type='gauss'):
        z = Variable(torch.randn(batchSize, nz)).cuda()
        return z

    # def test_forward(self, image, a2b=True):
    #     self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    #     if a2b:
    #         self.z_content = self.enc_c.forward_a(image)
    #         output = self.gen.forward_b(self.z_content, self.z_random)
    #     else:
    #         self.z_content = self.enc_c.forward_b(image)
    #         output = self.gen.forward_a(self.z_content, self.z_random)
    #     return output

    def forward(self):
        # get encoded z_c and z_s
        self.z_content_a, self.z_style_a = self.gen.encode(self.input_A, self.input_A_label)
        self.z_content_b, self.z_style_b = self.gen.encode(self.input_B, self.input_B_label)

        # get random z_a
        self.z_random = self.get_z_random(self.input_A.size(0), self.nz).cuda() #, 'gauss')
        self.fake_B_random = self.gen.decode(self.z_content_a,self.z_random,self.input_B_label)
        self.z_content_a_prime = self.gen.enc_content(self.fake_B_random)
        self.z_style_prime = self.gen.enc_style(self.fake_B_random,self.input_B_label)
        # Cycle reconstruction
        self.fake_A_cyc_recon = self.gen.decode(self.z_content_a_prime, self.z_style_a, self.input_A_label)

        # Reconstruction
        self.fake_A_recon = self.gen.decode(self.z_content_a,self.z_style_a, self.input_A_label)

        # # Style transfer
        # self.fake_AB = self.gen.decode(self.z_content_a,self.z_style_b,self.input_B_label)

        # for display
        self.image_display = torch.cat((self.input_A.detach().cpu(), self.input_B.detach().cpu(), self.fake_A_recon.detach().cpu(),\
                                        self.fake_A_cyc_recon.detach().cpu(), self.fake_B_random.detach().cpu() ), dim=0)



    def update_D(self, simgs,timgs,sl_onehot,tl_onehot,t):
        self.input_A = simgs
        self.input_B = timgs
        self.input_A_label = sl_onehot
        self.input_B_label = tl_onehot
        self.input_B_label_1 = t
        self.forward()

        # update discriminator
        self.dis_opt.zero_grad()
        Tensor = torch.cuda.FloatTensor
        self.valid = Variable(Tensor(self.input_A.shape[0], 1).fill_(1.0), requires_grad=False)
        self.fake = Variable(Tensor(self.input_A.shape[0], 1).fill_(0.0), requires_grad=False)

        # Compute loss with real images.
        out_src, out_cls = self.dis(self.input_B, self.input_B_label_1)
        #print(out_src) #,out_cls)
        print(out_src.shape, out_cls.shape)
        self.loss_dis_real = self.adversarial_loss(out_src,self.valid)         #   print(out_cls.shape,t.squeeze().shape)
        self.loss_dis_cls = self.classification_loss(out_cls,self.input_B_label_1.view(-1))    #torch.nn.functional.cross_entropy(out_cls, t2)#, print(d_loss_real,d_loss_cls)

        # Compute loss with fake (generated) images.
        out_src, out_cls = self.dis(self.fake_B_random.detach(),self.input_B_label_1)
        self.loss_dis_fake = self.adversarial_loss(out_src,self.fake)

        if printout:
            print('d_loss_real {}, d_loss_fake {} d_loss_cls {}'.format(self.loss_dis_real,self.loss_dis_fake,self.loss_dis_cls))

        # Backward and optimize.
        self.loss_dis_total = self.loss_dis_real + self.loss_dis_fake  + self.lambda_cls * self.loss_dis_cls
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_E(self):
        self.gen_opt.zero_grad()
        # Adversarial loss and Domain classification loss
        out_src, out_cls = self.dis(self.fake_B_random, self.input_B_label_1)
        self.loss_gan_fake = self.adversarial_loss(out_src,self.valid)
        self.loss_gan_cls = self.classification_loss(out_cls,self.input_B_label_1.view(-1))

        # Cycle consistence loss
        x_cyc_recons = self.gen.decode(self.z_content_a_prime,self.z_style_a,self.input_A_label)
        self.loss_gen_cyc = self.recon_criterion(self.input_A, x_cyc_recons)  #torch.mean(torch.abs(input - target)

        # Image reconstruction #Target-to-original domain.
        x_reconst = self.gen.decode(self.z_content_a,self.z_style_a,self.input_A_label)
        self.loss_gen_rec = self.recon_criterion(self.input_A, x_reconst)

        # Latent reconstruction
        # Content reconstruction
        self.loss_crec = self.recon_criterion(self.z_content_a, self.z_content_a_prime)
        # style reconstruction
        self.loss_srec = self.recon_criterion(self.z_random, self.z_style_prime)

        if printout:
            print('g_loss_fake {}, g_loss_cls {}'.format(self.loss_gan_fake,self.loss_gan_cls))
            print('loss_gen_cyc {}'.format(self.loss_gen_cyc))
            print('loss_gen_rec {}'.format(self.loss_gen_rec))
            print('loss_crec {}, loss_srec {}'.format(self.loss_crec,self.loss_srec))

        # Backward and optimize generator.
        self.loss_gen_total = self.loss_gan_fake + self.lambda_cls * self.loss_gan_cls + \
                              self.lambda_rec * self.loss_gen_rec + self.lambda_cyc * self.loss_gen_cyc + \
                              self.lambda_latent * (self.loss_srec + self.loss_crec)

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.dis.load_state_dict(checkpoint['dis'])

        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it'],checkpoint['lr']


    def save(self, filename, ep, total_it,lr):
        state = {
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'dis_opt': self.dis_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'lr': lr
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.input_A).detach()
        # images_b = self.normalize_image(self.input_B).detach()
        # images_a1 = self.normalize_image(self.fake_A_recon).detach()
        # images_a2 = self.normalize_image(self.fake_A_cyc_recon).detach()
        images_b1 = self.normalize_image(self.fake_B_random).detach()
        # return torch.cat((images_a, images_b, images_a1, images_a2, images_b1 ), dim=0)
        return torch.cat((images_a, images_b1), dim=0), self.input_B_label_1

    def normalize_image(self, x):
        return x[:,0:3,:,:]


    # def resume(self, checkpoint_dir, hyperparameters):
    #     # Load generators
    #     last_model_name = get_model_list(checkpoint_dir, "gen")
    #     state_dict = torch.load(last_model_name)
    #     self.gen.load_state_dict(state_dict)
    #     iterations = int(last_model_name[-11:-3])
    #     # Load discriminators
    #     last_model_name = get_model_list(checkpoint_dir, "dis")
    #     state_dict = torch.load(last_model_name)
    #     self.dis.load_state_dict(state_dict)
    #
    #     # Load optimizers
    #     state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
    #     self.dis_opt.load_state_dict(state_dict['dis'])
    #     self.gen_opt.load_state_dict(state_dict['gen'])
    #     # # Reinitilize schedulers
    #     # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
    #     # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
    #     print('Resume from iteration %d' % iterations)
    #     return iterations


    # def save(self, snapshot_dir, iterations):
    #     # Save generators, discriminators, and optimizers
    #     gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
    #     dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
    #     opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
    #     torch.save(self.gen.state_dict(), gen_name)
    #     torch.save(self.dis.state_dict(), dis_name)
    #     torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
    #


    # def gen_update(self, x,y,s,t):
    #     self.gen_opt.zero_grad()
    #     #prepare input data
    #     simgs = x.detach().cuda()            #== source img
    #     timgs = y.detach().cuda()             #== target img
    #     sl_onehot = self.label2onehot(s,self.opt.num_domains).cuda()  #== onehot source label
    #     tl_onehot= self.label2onehot(t,self.opt.num_domains).cuda()   #== onehot target label
    #     t = t.cuda()
    #
    #     Tensor = torch.cuda.FloatTensor
    #     valid = Variable(Tensor(simgs.shape[0], 1).fill_(1.0), requires_grad=False)
    #     z = Variable(torch.randn(simgs.shape[0],self.opt.style_dim)).cuda()
    #
    #     #encode
    #     content_s, style_s = self.gen.encode(simgs,sl_onehot)
    #
    #     # Adversarial loss and Domain classification loss
    #     timgs_out = self.gen.decode(content_s,z,tl_onehot)
    #     out_src, out_cls = self.dis(timgs_out,t)
    #     self.loss_gan_fake = self.adversarial_loss(out_src,valid)
    #     self.loss_gan_cls = self.classification_loss(out_cls,t.squeeze())
    #     if printout:
    #         print('g_loss_fake {}, g_loss_cls {}'.format(self.loss_gan_fake,self.loss_gan_cls))
    #
    #     # Cycle consistence loss
    #     Cprime_out = self.gen.enc_content(timgs_out)
    #     x_cyc_recons = self.gen.decode(Cprime_out,style_s,sl_onehot)
    #     self.loss_gen_cyc = self.recon_criterion(simgs, x_cyc_recons)  #torch.mean(torch.abs(input - target)
    #     if printout:
    #         print('loss_gen_cyc {}'.format(self.loss_gen_cyc))
    #
    #     # Image reconstruction #Target-to-original domain.
    #     x_reconst = self.gen.decode(content_s,style_s,sl_onehot)
    #     self.loss_gen_rec = self.recon_criterion(simgs, x_reconst)
    #     if printout:
    #         print('loss_gen_rec {}'.format(self.loss_gen_rec))
    #
    #     # Latent reconstruction
    #     # Content reconstruction
    #     self.loss_crec = self.recon_criterion(content_s, Cprime_out)
    #     # style reconstruction
    #     Sprime_out = self.gen.enc_style(timgs_out,tl_onehot)
    #     self.loss_srec = self.recon_criterion(z, Sprime_out)
    #     if printout:
    #         print('loss_crec {}, loss_srec {}'.format(self.loss_crec,self.loss_srec))
    #     #
    #     # Backward and optimize generator.
    #     self.loss_gen_total = self.loss_gan_fake + self.lambda_cls * self.loss_gan_cls + \
    #                           self.lambda_rec * self.loss_gen_rec + self.lambda_cyc * self.loss_gen_cyc + \
    #                           self.lambda_latent * (self.loss_srec + self.loss_crec)
    #
    #     self.loss_gen_total.backward() #retain_graph=True)
    #     self.gen_opt.step()
    #
    #
    # def dis_update(self, x,y,s,t):
    #     self.reset_grad()
    #     #prepare input data
    #     simgs = x.detach().cuda()            #== source img
    #     timgs = y.detach().cuda()             #== target img
    #     sl_onehot = self.label2onehot(s,self.opt.num_domains).cuda()  #== onehot source label
    #     tl_onehot= self.label2onehot(t,self.opt.num_domains).cuda()   #== onehot target label
    #     t = t.cuda()
    #     Tensor = torch.cuda.FloatTensor
    #     valid = Variable(Tensor(simgs.shape[0], 1).fill_(1.0), requires_grad=False)
    #     fake = Variable(Tensor(simgs.shape[0], 1).fill_(0.0), requires_grad=False)
    #     # Compute loss with real images.
    #     out_src, out_cls = self.dis(timgs,t)                          #print(out_src) #,out_cls)        #print(out_src.shape, out_cls.shape)
    #     self.loss_dis_real = self.adversarial_loss(out_src,valid)         #   print(out_cls.shape,t.squeeze().shape)
    #     self.loss_dis_cls = self.classification_loss(out_cls,t.squeeze())    #torch.nn.functional.cross_entropy(out_cls, t2)#, print(d_loss_real,d_loss_cls)
    #
    #     # Compute loss with fake (generated) images.
    #     content_s = self.gen.enc_content(simgs)                                      #content output (1,256*64*64)
    #     z = Variable(torch.randn(simgs.shape[0],self.opt.style_dim)).cuda()        # print(z.shape) (8)
    #     timgs_out = self.gen.decode(content_s,z,tl_onehot)                          #Generator output  # (1,3,256*256)
    #     out_src, out_cls = self.dis(timgs_out.detach(),t)
    #     self.loss_dis_fake = self.adversarial_loss(out_src,fake)
    #     if printout:
    #         print('d_loss_real {}, d_loss_fake {} d_loss_cls {}'.format(self.loss_dis_real,self.loss_dis_fake,self.loss_dis_cls))
    #
    #     # Backward and optimize.
    #     self.loss_dis_total = self.loss_dis_real + self.loss_dis_fake  + self.lambda_cls * self.loss_dis_cls
    #     self.loss_dis_total.backward()
    #     self.dis_opt.step()




    #

    # def sample(self, x_a, x_b):
    #     self.eval()
    #     s_a1 = Variable(self.s_a)
    #     s_b1 = Variable(self.s_b)
    #     s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
    #     s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
    #     x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
    #     x_ab, x_ba =[],[]
    #     for i in range(x_a.size(0)):
    #         c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
    #         c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
    #         x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
    #         x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
    #         x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
    #         x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
    #         x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
    #         x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
    #
    #         x_ab.append(self.gen_b.decode(c_a,s_b_fake))
    #         x_ba.append(self.gen_a.decode(c_b,s_a_fake))
    #     x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
    #     x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
    #     x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
    #     x_ab , x_ba = torch.cat(x_ab) , torch.cat(x_ba)
    #     self.train()
    #     return x_a, x_a_recon, x_ab1, x_ab2, x_ab, x_b, x_b_recon, x_ba1, x_ba2,x_ba
    #

    # def update_learning_rate(self):
    #     if self.dis_scheduler is not None:
    #         self.dis_scheduler.step()
    #     if self.gen_scheduler is not None:
    #         self.gen_scheduler.step()
    #

    # def forward(self, simgs, timgs, sl_onehot,tl_onehot):
    #     ''' tranfor a to b , b to a'''
    #     self.eval()
    #     # z = Variable(torch.randn(simgs.shape[0],self.opt.style_dim)).cuda()
    #     content_s, style_s = self.gen.encode(simgs,sl_onehot)    #S1 = self.SE(simgs,sl_onehot)  C1 = self.CE(simgs) S2 = self.SE(timgs,tl_onehot) C2 = self.CE(timgs)
    #     content_t, style_t = self.gen.encode(timgs,tl_onehot)
    #     S2T = self.G(content_s,style_t,tl_onehot)
    #     T2S = self.G(content_t,style_s,sl_onehot)
    #     self.train()
    #     return simgs,timgs,S2T, T2S