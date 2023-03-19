from numpy.core import numeric
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .resnet_models import resnet18, resnet50


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Identity(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class OnePlus(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()

    def forward(self, x):
        return F.softplus(x, beta=1)


class BWtoRGB(nn.Module):
    def __init__(self):
        super(BWtoRGB, self).__init__()

    def forward(self, x):
        assert len(list(x.size())) == 4
        chans = x.size(1)
        if chans < 3:
            return torch.cat([x, x, x], 1)
        else:
            return x


class MaskedConv2d(nn.Conv2d):
    ''' from jmtomczak's github '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class GatedConv2d(nn.Module):
    '''from jmtomczak's github '''
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels,
                           output_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation)
        self.g = nn.Conv2d(input_channels,
                           output_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class GatedConvTranspose2d(nn.Module):
    ''' from jmtomczak's github'''
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels,
                                    output_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels,
                                    output_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class EarlyStopping(object):
    def __init__(self,
                 model,
                 max_steps=10,
                 burn_in_interval=None,
                 save_best=True):
        self.max_steps = max_steps
        self.model = model
        self.save_best = save_best
        self.burn_in_interval = burn_in_interval

        self.loss = 0.0
        self.iteration = 0
        self.stopping_step = 0
        self.best_loss = np.inf

    def restore(self):
        self.model.load()

    def __call__(self, loss):
        if self.burn_in_interval is not None and self.iteration < self.burn_in_interval:
            ''' don't save the model until the burn-in-interval has been exceeded'''
            self.iteration += 1
            return False

        if (loss < self.best_loss):
            self.stopping_step = 0
            self.best_loss = loss
            if self.save_best:
                print("early stop saving best;  current_loss:{} | iter: {}".
                      format(loss, self.iteration))
                self.model.save(overwrite=True)
        else:
            self.stopping_step += 1

        is_early_stop = False
        if self.stopping_step >= self.max_steps:
            print(
                "Early stopping is triggered;  current_loss:{} --> best_loss:{} | iter: {}"
                .format(loss, self.best_loss, self.iteration))
            is_early_stop = True

        self.iteration += 1
        return is_early_stop


def flatten_layers(model, base_index=0):
    layers = []
    for l in model.children():
        if isinstance(l, nn.Sequential):
            sub_layers, base_index = flatten_layers(l, base_index)
            layers.extend(sub_layers)
        else:
            layers.append(('layer_%d' % base_index, l))
            base_index += 1

    return layers, base_index


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("initializing ", m, " with xavier init")
            nn.init.xavier_uniform(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod)

    return module


class IsotropicGaussian(nn.Module):
    def __init__(self, mu, logvar, use_cuda=False):
        super(IsotropicGaussian, self).__init__()
        self.mu = mu
        self.dims = self.mu.size()[1]
        self.logvar = logvar
        self.use_cuda = use_cuda

    def mean(self):
        return self.mu

    def log_var(self):
        return self.logvar

    def var(self):
        return self.logvar.exp()

    def update_add(self, mu_update, logvar_update, eps=1e-9):
        self.mu += mu_update
        self.logvar = torch.log(self.logvar.exp() * logvar_update.exp() + eps)

    def sample(self, mu, logvar):
        eps = Variable(float_type(self.use_cuda)(self.logvar.size()).normal_())
        return mu + logvar.exp() * eps

    def forward(self, logits, return_mean=False):
        ''' If return_mean is true then returns mean instead of sample '''
        mu, var = _divide_logits(logits)
        logvar = F.softplus(var)
        self.update_add(mu, logvar)
        return self.mu if return_mean else self.sample(self.mu, self.logvar)


class Convolutional(nn.Module):
    def __init__(
        self,
        input_size,
        layer_maps=[32, 64, 128, 64, 32],
        # kernel_sizes=[3, 3, 3, 3, 3],
        kernel_sizes=[1, 1, 1, 1, 1],
        activation_fn=nn.ELU,
        use_dropout=False,
        use_bn=False,
        use_in=False,
        use_wn=False,
        ngpu=1):
        super(Convolutional, self).__init__()
        ''' input_size = 2d or 3d'''

        assert len(kernel_sizes) == len(layer_maps)

        # Parameters pass into layer
        self.input_size = input_size
        self.is_color = input_size[-1] > 1
        self.layer_maps = [3 if self.is_color else 1] \
                          + layer_maps \
                          + [3 if self.is_color else 1]
        self.kernel_sizes = [1] + kernel_sizes + [1]
        self.activation = activation_fn
        self.use_bn = use_bn
        self.use_in = use_in
        self.use_wn = use_wn
        self.ngpu = ngpu
        self.use_dropout = use_dropout

        # Are we using a normalization scheme?
        self.use_norm = bool(use_bn or use_in)

        # Build our model as a sequential layer
        self.net = self._build_layers()
        self.add_module('network', self.net)
        #self = init_weights(self)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output

    def cuda(self, device_id=None):
        super(Convolutional, self).cuda(device_id)
        self.net = self.net.cuda()
        return self

    def get_info(self):
        return self.net.modules()

    def get_sizing(self):
        return str(self.sizes)

    def _add_normalization(self, num_features):
        if self.use_bn:
            return nn.BatchNorm2d(num_features)
        elif self.use_in:
            return nn.InstanceNorm2d(num_features)

    def _add_dropout(self):
        if self.use_dropout:
            return nn.AlphaDropout()

    def _build_layers(self):
        '''Conv/FC --> BN --> Activ --> Dropout'''
        'Conv2d maps (N, C_{in}, H, W) --> (N, C_{out}, H_{out}, W_{out})'
        # if not self.is_color:
        #     layers = [('flatten', View([-1, 1] + self.input_size))]
        # else:
        layers = []

        for i, (in_channels, out_channels) in enumerate(
                zip(self.layer_maps[0:-1], self.layer_maps[1:-1])):

            l = nn.Conv2d(in_channels,
                          out_channels,
                          self.kernel_sizes[i],
                          padding=0)
            if self.use_wn:
                layers.append(('conv_%d' % i, nn.utils.weight_norm(l)))
            else:
                layers.append(('conv_%d' % i, l))

            if self.use_norm:  # add normalization
                layers.append(
                    ('norm_%d' % i, self._add_normalization(out_channels)))

            layers.append(('activ_%d' % i, self.activation()))

            if self.use_dropout:  # add dropout
                layers.append(('dropout_%d' % i, self._add_dropout()))

        l_f = nn.Conv2d(self.layer_maps[-2],
                        self.layer_maps[-1],
                        self.kernel_sizes[-1],
                        padding=0)
        layers.append(('conv_proj', l_f))
        return nn.Sequential(OrderedDict(layers))


class UpsampleConvLayer(torch.nn.Module):
    '''Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/ '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,
                                                    mode='bilinear')

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class Dense(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 layer_sizes,
                 activation_fn,
                 use_dropout=False,
                 use_bn=False,
                 use_in=False,
                 use_wn=False,
                 ngpu=1):
        super(Dense, self).__init__()

        # Parameters pass into layer
        self.input_size = input_size
        self.latent_size = latent_size
        self.layer_sizes = [input_size] + layer_sizes
        self.activation = activation_fn
        self.use_bn = use_bn
        self.use_in = use_in
        self.use_wn = use_wn
        self.ngpu = ngpu
        self.use_dropout = use_dropout

        # Are we using a normalization scheme?
        self.use_norm = bool(use_bn or use_in)

        # Build our model as a sequential layer
        self.net = self._build_layers()
        self.add_module('network', self.net)
        #self = init_weights(self)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output

    def cuda(self, device_id=None):
        super(Dense, self).cuda(device_id)
        self.net = self.net.cuda()
        return self

    def get_info(self):
        return self.net.modules()

    def get_sizing(self):
        return str(self.sizes)

    def _add_normalization(self, num_features):
        if self.use_bn:
            return nn.BatchNorm1d(num_features)
        elif self.use_in:
            return nn.InstanceNorm1d(num_features)

    def _add_dropout(self):
        if self.use_dropout:
            return nn.AlphaDropout()

    def _build_layers(self):
        '''Conv/FC --> BN --> Activ --> Dropout'''
        layers = [('flatten', View([-1, self.input_size]))]
        for i, (input_size, output_size) in enumerate(
                zip(self.layer_sizes, self.layer_sizes[1:])):
            if self.use_wn:
                layers.append(
                    ('linear_%d' % i,
                     nn.utils.weight_norm(nn.Linear(input_size, output_size))))
            else:
                layers.append(
                    ('linear_%d' % i, nn.Linear(input_size, output_size)))

            if self.use_norm:  # add normalization
                layers.append(
                    ('norm_%d' % i, self._add_normalization(output_size)))

            layers.append(('activ_%d' % i, self.activation()))

            if self.use_dropout:  # add dropout
                layers.append(('dropout_%d' % i, self._add_dropout()))

        layers.append(
            ('linear_proj', nn.Linear(self.layer_sizes[-1], self.latent_size)))
        return nn.Sequential(OrderedDict(layers))


def build_image_downsampler(img_shp, new_shp, stride=[3, 3], padding=[0, 0]):
    '''Takes a tensor and returns a downsampling operator'''
    equality_test = np.asarray(img_shp) == np.asarray(new_shp)
    if equality_test.all():
        return Identity()

    height = img_shp[0]
    width = img_shp[1]
    new_height = new_shp[0]
    new_width = new_shp[1]

    # calculate the width and height by inverting the equations from:
    # http://pytorch.org/docs/master/nn.html?highlight=avgpool2d#torch.nn.AvgPool2d
    kernel_width = -1 * ((new_width - 1) * stride[1] - width - 2 * padding[1])
    kernel_height = -1 * (
        (new_height - 1) * stride[0] - height - 2 * padding[0])
    print('kernel = ', kernel_height, 'x', kernel_width)
    assert kernel_height > 0
    assert kernel_width > 0

    return nn.AvgPool2d(kernel_size=(kernel_height, kernel_width),
                        stride=stride,
                        padding=padding)


def build_encoder(input_chans, output_dim, arch):
    if arch == 1:
        return nn.Sequential(
            nn.Conv2d(input_chans, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 1),  # B, 32,  1,  1
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            View((-1, 32 * 1 * 1)),  # B, 32
            nn.Linear(32, output_dim)  # B, z_dim*2
        )
    elif arch == 2:
        return nn.Sequential(
            nn.Conv2d(input_chans, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),  # B, 32,  1,  1
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 32
            nn.Linear(256, output_dim)  # B, z_dim*2
        )
    elif arch == 3:
        return nn.Sequential(
            nn.Conv2d(input_chans, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),  # B,  32,  16,  16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),  # B, 32,  1,  1
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 32
            nn.Linear(256, output_dim)  # B, z_dim*2
        )
    elif arch == 4:
        return nn.Sequential(
            resnet18(pretrained=False, input_chans=input_chans),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, output_dim))


def build_decoder(input_dim, output_chans, arch):
    if arch == 1:
        return nn.Sequential(
            nn.Linear(input_dim, 32),  # B, 256
            View((-1, 32, 1, 1)),  # B, 256,  1,  1
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4),  # B,  64,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_chans, 4, 2, 1)  # B, nc, 32, 32
        )
    elif arch == 2:
        return nn.Sequential(
            nn.Linear(input_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),  # B,  64,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_chans, 4, 2, 1)  # B, nc, 32, 32
        )
    elif arch == 3:
        return nn.Sequential(
            nn.Linear(input_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),  # B,  64,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),  # B,  64,  4,  4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),  # B,  64,  16,  16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_chans, 4, 2, 1)  # B, nc, 32, 32
        )
    elif arch == 4:
        return
