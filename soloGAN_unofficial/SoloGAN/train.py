
from options.train_options import TrainOptions
from trainer import SoloGAN
from dataset import dataset_multi
import torch.backends.cudnn as cudnn
import torch
from utils import label2onehot,generateTargetLabel
#from itertools import izip as zip
from saver import Saver


def main():
    # Load experiment setting
    opt = TrainOptions().parse()
    # For fast training.
    cudnn.benchmark = True

    print('\n--- load dataset ---')
    dataset = dataset_multi(opt,phase='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    # dataset = dataset_multi(opt,phase='test')
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

    # Setup model and data loader
    print('\n--- load model ---')
    model = SoloGAN(opt)
    model.cuda()

    lr = opt.lr

    if opt.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it,lr = model.resume(opt.resume)
        model.update_lr(lr)
        # model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opt)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opt.epoch):
        for it, (simgs,timgs, sourceID,targetID) in enumerate(train_loader):
            #print('epoch  {}  iteration {}'.format(ep,it))
            # Generate the target image and corresponding label
            # targetID =[]                #print('num of images in batch {} and its length of label{}'.format(len(images),len(sourceID)))
            # timages =[]
            # for i,(img,sl) in enumerate(zip(images,sourceID)):
            #     s = sl.numpy()[0]
            #     t = generateTargetLabel(s, opt.num_domains)
            #     targetID.append(torch.tensor([t]))
            #     timages.append(images[t])

            simgs = torch.stack([simgs[i].squeeze() for i in range(len(simgs))])
            slabel = torch.stack([sourceID[i] for i in range(len(sourceID))])
            timgs = torch.stack([timgs[i].squeeze() for i in range(len(timgs))])
            tlabel = torch.stack([targetID[i] for i in range(len(targetID))])
            tlabel = torch.tensor([[1]])

            #prepare input data
            simgs = simgs.detach().cuda()            #== source img
            timgs = timgs.detach().cuda()             #== target img
            sl_onehot = label2onehot(slabel,opt.num_domains).cuda()  #== onehot source label
            tl_onehot= label2onehot(tlabel,opt.num_domains).cuda()   #== onehot target label
            s= slabel.cuda()
            t = tlabel.cuda()

            print('simgs shape ',simgs.shape)
            print('timgs shape ',timgs.shape)
            print('slabel shape ',slabel.shape)
            print('tlabel shape ',tlabel.shape)
            # Update discriminator
            model.update_D(simgs,timgs,sl_onehot,tl_onehot,t)
            model.update_E()
            torch.cuda.synchronize()

            # save to display file   # write losses and images to tensorboard
            if not opt.no_display_img:
                saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(ep,total_it, model,lr)
                break

        # decay learning rate
        if ep > opt.niter:
            lr -= opt.lr / opt.niter_decay
            model.update_lr(lr)

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        # saver.write_model(ep, total_it, model,lr)
    return


if __name__ == '__main__':
    main()