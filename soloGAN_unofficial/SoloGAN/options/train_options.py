from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
        #self.parser.add_argument('--resume', type=str, default='./results/trial/00009.pth', help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--log_iter', type=int, default =1,help='How often do you want to log the training stats')
        self.parser.add_argument('--max_iter', type=int, default =200000,help='maximum number of iterations')
        self.parser.add_argument('--image_save_iter', type=int, default=200, help='How often do you want to save output images during training')
        self.parser.add_argument('--display_size', type=int, default=4, help='How many images do you want to display each time ')


        self.parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')
        # learning rate
        self.parser.add_argument('--lr', type=float, default=0, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')

        # lambda parameters
        self.parser.add_argument('--lambda_cyc', type=float, default=25.0, help='weight for cycle consistency')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for identity consistency')
        self.parser.add_argument('--lambda_cls', type=float, default=1.0, help='weight for class')
        self.parser.add_argument('--lambda_latent', type=float, default=1.0, help='weight for latents')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='/content/drive/MyDrive/soloGAN/logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='/content/drive/MyDrive/soloGAN/results', help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=1, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')
        self.isTrain = True
