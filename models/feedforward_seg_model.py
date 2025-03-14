import torch
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output parameters
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample)
        if self.use_cuda: 
            self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            if kwargs.get('verbose', True):
                print('✅ Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('📌 Scheduler is added for optimizer {0}'.format(optimizer))

    def set_input(self, *inputs):
        for idx, _input in enumerate(inputs):
            if idx == 0:
                # ✅ Ensure shape is [B, 1, D, H, W] (not [B, 2, D, H, W])
                if _input.shape[1] != 1:
                    _input = _input[:, :1, :, :, :]  # Keep only 1 channel
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                # Ensure target shape matches input shape, removing extra channel if present
                if _input.dim() == 5:  # Shape [B, 1, D, H, W]
                    self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                elif _input.dim() == 6:  # Shape [B, 1, 1, D, H, W]
                    self.target = Variable(_input.squeeze(2).cuda()) if self.use_cuda else Variable(_input.squeeze(2))
                else:
                    raise ValueError(f"Unexpected target shape: {_input.shape}")

        print(f"✅ Debug: Final Input Shape = {self.input.shape}, Final Target Shape = {self.target.shape}")
        assert self.input.size() == self.target.size(), \
            f"❌ Shape Mismatch: Input {self.input.size()} vs Target {self.target.size()}"
        
    def forward(self, split):
        """
        Forward pass through the network.
        """
        # ✅ Debug Print Before Forward Pass
        print(f"🚀 Forward Pass: Model Input Shape = {self.input.shape}")

        if split == 'train':
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            self.prediction = self.net(Variable(self.input, volatile=True))
            # Apply a softmax and return a segmentation map
            self.logits = self.net.apply_argmax_softmax(self.prediction)
            self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """
        Computes the loss and performs backpropagation.
        """
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        """
        Optimizes the model parameters by performing a forward and backward pass.
        """
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    def optimize_parameters_accumulate_grd(self, iteration):
        """
        This function updates the network parameters every "accumulate_iters" iterations.
        """
        accumulate_iters = 2
        if iteration == 0: 
            self.optimizer_S.zero_grad()

        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        """
        Runs inference on test data.
        """
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        """
        Runs validation on test data.
        """
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        """
        Computes segmentation metrics like accuracy and IoU.
        """
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        """
        Returns current segmentation loss.
        """
        return OrderedDict([('Seg_Loss', self.loss_S.item())])

    def get_current_visuals(self):
        """
        Returns input image and segmentation output.
        """
        inp_img = util.tensor2im(self.input, 'img')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        """
        Extracts feature maps from a given network layer.
        """
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    def get_fp_bp_time(self, size=None):
        """
        Returns the forward pass and backward pass time of the model.
        """
        if size is None:
            size = (1, 1, 160, 160, 96)  # Updated to match expected shape

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        """
        Saves the trained model.
        """
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)