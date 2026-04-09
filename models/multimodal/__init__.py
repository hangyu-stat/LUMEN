import functools
from ..attentions import get_attention_module
from .block import BasicBlock, BottleNect, InvertedResidualBlock
from .resnet import ResNet18, ResNet34, ResNet50
from .unet_model import UNet
from .CheXNet import DenseNet121, LightDenseNet121
from .vgg import VGG16_bn
from .mobilenetv2 import MobileNetV2Wrapper
from.ViT import ViT
import torch
import torch.nn as nn
from ..attentions.ca_module import ca_module
from ..attentions.attention_module import attention_module
from linformer import Linformer


model_dict = {
    "resnet18": ResNet18, 
    "resnet34": ResNet34, 
    "resnet50": ResNet50,
    "vgg16_bn": VGG16_bn,
    "mobilenetv2": MobileNetV2Wrapper,
    'unet': UNet,
    'densenet121': DenseNet121,
    'vit': ViT,
    'lightdensenet121': LightDenseNet121
}

CHEXNET_CKPT_PATH = './models/CheXNetCKPT/CheXNet.pth.tar'


def get_block(block_type="basic"):

    block_type = block_type.lower()

    if block_type == "basic":
        b = BasicBlock
    elif block_type == "bottlenect":
        b = BottleNect
    elif block_type == "ivrd":
        b = InvertedResidualBlock
    elif block_type == "vgg":
        b = "vgg"
    else:
        raise NotImplementedError(
            'block [%s] is not found for dataset [%s]' % block_type)
    return b


def create_net(args):
    net = None

    block_module = get_block(args.block_type)
    attention_module = get_attention_module(args.attention_type)

    if args.attention_type == "se" or args.attention_type == "cbam" or args.attention_type == "gc":
        attention_module = functools.partial(
            attention_module, reduction=float(args.attention_param))
    elif args.attention_type == "eca":
        attention_module = functools.partial(
            attention_module, k_size=int(args.attention_param))
    elif args.attention_type == "wa":
        attention_module = functools.partial(
            attention_module, wavename=args.attention_param)

    net = model_dict[args.arch.lower()](
        block=block_module,
        attention_module=attention_module,
        num_filters=args.num_base_filters
    )

    return net


class TransformerAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, dim_feedforward=512, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, 512, 768] from BERT
        x = self.linear_in(x)  # -> [B, 512, 128]
        x = self.transformer(x)  # -> [B, 512, 128]
        return x


class MultimodalNet(nn.Module):
    def __init__(self, feature_extractor, image_feature_dim, input_dim_x, output_dim_y, num_slices=20, report_flag=False, report_sentences_num=30, reasoning_flag=False, reasoning_sentences_num=30, ca_num_heads=4, dp_rate=0.2, classification=False):
        super(MultimodalNet, self).__init__()
        self.num_slices = num_slices
        self.report_sentences_num = report_sentences_num
        self.reasoning_sentences_num = reasoning_sentences_num
        self.dp_rate = dp_rate
        self.report_flag = report_flag
        self.reasoning_flag = reasoning_flag
        self.feature_extractor = feature_extractor  # 图像特征提取网络
        self.image_feature_dim = image_feature_dim
        self.res_num = output_dim_y
        self.classification = classification
        if self.report_flag:
            self.linear_report = nn.Sequential(
                nn.LayerNorm(769),
                nn.GELU(),
                nn.Linear(769, image_feature_dim),
                nn.LayerNorm(image_feature_dim),
                nn.GELU(),
                nn.Linear(image_feature_dim, 32),
                nn.LayerNorm(32),
                nn.GELU()
            )
        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.ln_img = nn.LayerNorm(self.image_feature_dim)  # need to fix
        self.ln_cov = nn.LayerNorm(16)

        self.linear_cov = nn.Linear(input_dim_x, 16)  # 处理 p 维协变量 x
        self.img_self_att = attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads)

        if self.reasoning_flag:
            self.reason_token_num = 512
            self.linear_reasoning = nn.Sequential(
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, image_feature_dim),
                nn.LayerNorm(image_feature_dim),
                nn.GELU()
            )
            self.img2reason_CAs = nn.ModuleList()
            self.reason2img_CAs = nn.ModuleList()
            self.CA_ln_imgs = nn.ModuleList()
            for i in range(self.reasoning_sentences_num):
                self.img2reason_CAs.append(
                    attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads))
                self.reason2img_CAs.append(
                    attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads))
                self.CA_ln_imgs.append(nn.LayerNorm(self.image_feature_dim))

        self.mlp_layers = nn.ModuleList()
        self.mlp_input_dim = image_feature_dim + 16
        if self.report_flag:
            self.mlp_input_dim += 32
        if self.reasoning_flag:
            self.conv1d = nn.Conv1d(in_channels=int(self.num_slices * (self.reasoning_sentences_num + 1)),
                                    out_channels=1, kernel_size=1)
        else:
            self.conv1d = nn.Conv1d(in_channels=int(self.num_slices), out_channels=1, kernel_size=1)

        if self.classification:
            for i in range(output_dim_y):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, 1),
                    nn.Sigmoid()
                ))

        else:
            for i in range(1):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, output_dim_y)
                ))

    def forward(self, image, cov, report=None, reasoning=None, reasoning_mask=None):
        '''
        1. When report_flag=False and reasoning_flag=False: Pure Image Net.
        2. When report_flag=True and reasoning_flag=False: Image + Report Net.
        3. When report_flag=False and reasoning_flag=True: Image + Reasoning Net.
        * report_flag and reasoning_flag should not be True at the same time because  CT report and reasoning contain a large amount of overlapping text.
        '''
        # image feature extraction
        batch_size = image.shape[0]
        images = image.view(-1, 1, image.shape[2], image.shape[3])
        images_features = self.feature_extractor(images).view(batch_size, self.num_slices, self.image_feature_dim)
        images_features = self.ln_img(images_features)
        images_features, _ = self.img_self_att(images_features, images_features, images_features)  # b*10*128
        # cov feature extraction
        cov_features = self.ln_cov(self.linear_cov(cov))

        if self.report_flag:
            report_features = self.linear_report(report)  # b*1*32
            pooled_report_feature = report_features.squeeze(1)  # b*32

        if self.reasoning_flag:

            reasoning_features = self.linear_reasoning(reasoning)  # b*3*512*128

            images_features_cat = images_features
            for i in range(self.reasoning_sentences_num):
                # consider CA + CA bi-directional
                reasoning_features_up, _ = self.reason2img_CAs[i](reasoning_features[:, i, :, :].view(batch_size, -1, self.image_feature_dim), images_features, images_features)
                key_reasoning_mask = reasoning_mask[:, i, :].view(batch_size, -1)
                key_reasoning_mask = key_reasoning_mask == 0
                images_features_up, _ = self.img2reason_CAs[i](images_features, reasoning_features_up, reasoning_features_up, key_padding_mask=key_reasoning_mask)
                images_features_up = self.CA_ln_imgs[i](images_features_up + images_features)
                images_features_cat = torch.cat((images_features_cat, images_features_up), dim=1)

            pooled_images_feature = self.conv1d(images_features_cat).squeeze(1)  # b*128

            mm_features = torch.cat((pooled_images_feature, cov_features), dim=1)  # b*(128+16)

            output0 = self.mlp_layers[0](mm_features)
            output1 = self.mlp_layers[1](mm_features)
            output2 = self.mlp_layers[2](mm_features)

        else:
            pooled_images_feature = self.conv1d(images_features).squeeze(1)  # b*128
            if self.report_flag:
                mm_features = torch.cat((pooled_images_feature, pooled_report_feature, cov_features), dim=1)
            else:
                mm_features = torch.cat((pooled_images_feature, cov_features), dim=1)
            # output = self.mlp_layers[0](mm_features)

            output0 = self.mlp_layers[0](mm_features)
            output1 = self.mlp_layers[1](mm_features)
            output2 = self.mlp_layers[2](mm_features)

        output = torch.cat((output0, output1, output2), dim=1)

        return output


class TextmodalNet(nn.Module):
    def __init__(self, text_feature_dim, input_dim_x, output_dim_y, report_flag=False, report_sentences_num=30, reasoning_flag=False, reasoning_sentences_num=30, dp_rate=0.2, classification=False):
        super(TextmodalNet, self).__init__()
        self.report_sentences_num = report_sentences_num  # token number
        self.reasoning_sentences_num = reasoning_sentences_num  # token number
        self.dp_rate = dp_rate
        self.report_flag = report_flag
        self.reasoning_flag = reasoning_flag
        self.text_feature_dim = text_feature_dim
        self.res_num = output_dim_y
        self.classification = classification
        if self.report_flag:
            self.linear_report = nn.Sequential(
                nn.LayerNorm(769),
                nn.GELU(),
                nn.Linear(769, text_feature_dim),
                nn.LayerNorm(text_feature_dim),
                nn.GELU(),
                nn.Linear(text_feature_dim, 32),
                nn.LayerNorm(32),
                nn.GELU()
            )

        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.ln_cov = nn.LayerNorm(16)

        self.linear_cov = nn.Linear(input_dim_x, 16)  # 处理 p 维协变量 x

        if self.reasoning_flag:
            self.linear_reasoning = nn.Sequential(
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, text_feature_dim),
                nn.LayerNorm(text_feature_dim),
                nn.GELU(),
                nn.Linear(text_feature_dim, 32),
                nn.LayerNorm(32),
                nn.GELU()
            )

        self.mlp_layers = nn.ModuleList()
        if self.report_flag and self.reasoning_flag:
            self.mlp_input_dim = 32 + 96 + 16
        else:
            if self.report_flag:
                self.mlp_input_dim = 32 + 16
            else:
                self.mlp_input_dim = 96 + 16

        if self.classification:
            for i in range(output_dim_y):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, 1),
                    nn.Sigmoid()
                ))

        else:
            for i in range(1):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, output_dim_y)
                ))

    def forward(self, cov, report=None, reasoning=None):
        batch_size = cov.shape[0]
        # cov feature extraction
        cov_features = self.ln_cov(self.linear_cov(cov))

        if self.report_flag:
            report_features = self.linear_report(report)  # b*1*32
            pooled_report_feature = report_features.squeeze(1)  # b*32

        if self.reasoning_flag:
            reasoning_features = reasoning[:, :, 0, :]
            reasoning_features = self.linear_reasoning(reasoning_features)  # b*3*32
            pooled_reasoning_features = reasoning_features.view(batch_size, -1)  # b*96

        if self.report_flag:
            mm_features = torch.cat((pooled_report_feature, cov_features), dim=1)
        else:
            mm_features = torch.cat((pooled_reasoning_features, cov_features), dim=1)

        output0 = self.mlp_layers[0](mm_features)
        output1 = self.mlp_layers[1](mm_features)
        output2 = self.mlp_layers[2](mm_features)
        output = torch.cat((output0, output1, output2), dim=1)
        return output


def create_multimodal_net(args):
    """
    创建多模态网络，包含图像特征提取网络和协变量处理模块，最后进行回归拟合。
    """
    if args.arch.lower() == 'densenet121':
        feature_extractor = model_dict[args.arch.lower()](out_size=args.image_feature_dim)
        checkpoint = torch.load(CHEXNET_CKPT_PATH)
        feature_extractor.load_state_dict(checkpoint['state_dict'], strict=False)
    elif args.arch.lower() == 'lightdensenet121':
        feature_extractor = model_dict[args.arch.lower()](out_size=args.image_feature_dim)

    # 定义完整的多模态网络
    # 实例化多模态网络
    multimodal_net = MultimodalNet(
        feature_extractor=feature_extractor,
        image_feature_dim=args.image_feature_dim,
        input_dim_x=args.cov_dim,  # 协变量的维度
        output_dim_y=args.res_dim,  # 响应变量的维度
        num_slices=args.CT_slice_num,
        report_flag=args.CT_report_flag,
        report_sentences_num=args.CT_report_sentence_num,
        reasoning_flag=args.reasoning_flag,
        reasoning_sentences_num=args.reasoning_sentence_num,
        ca_num_heads=args.CA_num_heads,
        dp_rate=args.dropout,
        classification=args.classification
    )

    return multimodal_net


def create_textmodal_net(args):
    textmodal_net = TextmodalNet(
        text_feature_dim=args.image_feature_dim,
        input_dim_x=args.cov_dim,  # 协变量的维度
        output_dim_y=args.res_dim,  # 响应变量的维度
        report_flag=args.CT_report_flag,
        report_sentences_num=args.CT_report_sentence_num,
        reasoning_flag=args.reasoning_flag,
        reasoning_sentences_num=args.reasoning_sentence_num,
        dp_rate=args.dropout,
        classification=args.classification
    )

    return textmodal_net