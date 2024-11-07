import torch
import torch.nn as nn
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .unimatch.utils import feature_add_position

from .unimatch_transformer import FeatureTransformer

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth


class BackboneUniMatch(torch.nn.Module):
    """docstring for BackboneUniMatch."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackboneUniMatch, self).__init__()
        self.feature_channels = feature_channels
        self.use_epipolar_trans = use_epipolar_trans

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = FeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            wo_cross_attn=no_cross_attn,
            wo_self_attn=no_self_attn,
        )
        # self.global_attn_fast = global_attn_fast
        # self.pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

        # used to provide pixel-algin gaussains for color related info
        self.upscale_factor = 1
        # if upscale_factor > 1:
        #     d_in = feature_channels
        #     self.upscaler = nn.ConvTranspose2d(
        #         d_in, d_in, upscale_factor, upscale_factor
        #     )
        #     self.upscale_refinement = nn.Sequential(
        #         nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
        #         nn.GELU(),
        #         nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
        #     )

    def normalize_img(self, img0, img1):
        # loaded images are in [0, 1.]
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        img0 = (img0 - mean) / std
        img1 = (img1 - mean) / std

        return img0, img1

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(
            concat
        )  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        """
        img0: range (0, 1.), shape: (b, v, c, h, w)
        """
        img0 = images[:, 0]
        img1 = images[:, 1]

        feature0_list, feature1_list = self.extract_feature(
            *self.normalize_img(img0, img1)
        )

        # assume only 1 scale for now
        feature0, feature1 = feature0_list[0], feature1_list[0]

        if return_cnn_features:
            cnn_features = torch.stack((feature0, feature1), dim=1)  # [B, V, C, H, W]

        if self.use_epipolar_trans:
            assert (
                epipolar_kwargs is not None
            ), "must provide camera params to apply epipolar transformer"
            epipolar_sampler = epipolar_kwargs["epipolar_sampler"]
            depth_encoding = epipolar_kwargs["depth_encoding"]

            features = torch.stack((feature0, feature1), dim=1)  # [B, V, C, H, W]
            extrinsics = epipolar_kwargs["extrinsics"]
            intrinsics = epipolar_kwargs["intrinsics"]
            near = epipolar_kwargs["near"]
            far = epipolar_kwargs["far"]
            # Get the samples used for epipolar attention.
            sampling = epipolar_sampler.forward(
                features, extrinsics, intrinsics, near, far
            )
            # similar to pixelsplat, use camera distance as position encoding
            # Compute positionally encoded depths for the features.
            collect = epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = depth_encoding(depths[..., None])
            target = sampling.features + depths
            source = features

            features = self.transformer(source, target, attn_type="epipolar")
        else:
            feature0, feature1 = feature_add_position(
                feature0, feature1, attn_splits, self.feature_channels
            )

            # Transformer
            # when local correlation with flow, reuse features after transformer, no need to recompute
            feature0, feature1 = self.transformer(
                feature0,
                feature1,
                attn_type="swin",
                attn_num_splits=attn_splits,
            )

            features = torch.stack((feature0, feature1), dim=1)  # (b, v, c, h, w)

        if self.upscale_factor > 1:
            b, v = features.shape[:2]
            features_up = self.upscaler(rearrange(features, "b v c h w -> (b v) c h w"))
            features_up = self.upscale_refinement(features_up) + features_up
            features_up = rearrange(features_up, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            features_up = None
        out_lists = [features, features_up]

        if return_cnn_features:
            # features_up is useless
            out_lists = [features, cnn_features]

        return out_lists
