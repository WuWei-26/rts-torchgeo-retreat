# rts_model_retreat.py

from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unetplusplus import UnetPlusPlus
from segmentation_models_pytorch.encoders.timm_universal import TimmUniversalEncoder
from torchgeo.models import ResNet50_Weights
import copy
import pytorch_lightning as pl

from rts_utils import (
    AREA_K_SIZE,
    AREA_STRIDE,
    AreaPoolLoss,
    LossWeightedAWing,
    LossWeightedMSE,
    pad_HW,
)

S2_2_L8 = {
    "SR_B1": 1,
    "SR_B2": 2,
    "SR_B3": 3,
    "SR_B4": 4,
    "SR_B5": 9,  # 8a, not 8
    "SR_B6": 12,  # not 11
    "SR_B7": 13,  # not 12
}
# band number start with 1
# BAND_IDX_S2 = list(np.array(itemgetter(*train_imgs.bands)(S2_2_L8)) - 1)

RETREAT_SCALE =  28.90128517150879 # global_max retreat = 28.90128517150879
# RETREAT_SCALE =  1.0 # dont scale

def resnet50_encoder(
    weights: Optional[ResNet50_Weights] = None,
    in_bands: List[str] = [],
    depth: int = 5,
    use_dem: bool = True,
    output_stride: int = 32,
    # band_idx: list = [],
    *args: Any,
    **kwargs: Any,
):
    """ResNet-50 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    .. versionchanged:: 0.4
       Switched to multi-weight support API.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`.

    Returns:
        A ResNet-50 model.
    """
    if weights:
        weights_in_channels = weights.meta["in_chans"]

    # model: ResNet = timm.create_model("resnet50", *args, **kwargs)
    # Encoder = resnet_encoders['resnet50']["encoder"]
    # params = resnet_encoders['resnet50']["params"]
    # # params.update(depth=depth)
    # encoder = Encoder(**params)

    # initialize encoder module
    encoder = TimmUniversalEncoder(
        name="resnet50",
        in_channels=weights_in_channels,
        depth=depth,
        output_stride=output_stride,
        pretrained=weights is not None,
        **kwargs,
    )
    # display(encoder.model)

    if weights:
        state_dict = weights.get_state_dict(progress=True)
        encoder.model.load_state_dict(state_dict=state_dict, strict=True)
        print(weights, " loaded")

    new_in_channels = len(in_bands)
    if use_dem:
        new_in_channels = new_in_channels + 1
    if weights_in_channels == new_in_channels:
        print("no need to adjust weights for input channel")
        return encoder

    for module in encoder.model.modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == weights_in_channels
        ):
            # print(module)
            break

    weight = module.weight.detach()
    # print(weight.shape)
    module.in_channels = new_in_channels

    new_weight = torch.Tensor(
        module.out_channels, new_in_channels // module.groups, *module.kernel_size
    )
    torch.nn.init.zeros_(new_weight)
    # print(new_weight.shape)

    # train_imgs.bands
    # for i in range(new_in_channels):
    #     new_weight[:, i] = weight[:, i % weights_in_channels]
    # band_idx = np.array(itemgetter(*train_imgs.bands)(S2_2_L8)) - 1
    print("input shape of target weight", new_weight.shape)
    print("input shape of loaded weight", weight.shape)

    if weights_in_channels == 3 and len(in_bands) == 3:
        # weights with RGB bands only
        selected_weight = weight
    else:
        # weights with ALL bands
        band_idx = list(np.array(itemgetter(*in_bands)(S2_2_L8)) - 1)
        # print(band_idx)
        selected_weight = weight[:, band_idx]
    if not use_dem:  # new_in_channels == len(band_idx):
        # no using dem, landsat_bands only
        new_weight = selected_weight
        # new_weight = new_weight * (weights_in_channels / new_in_channels)
    # elif new_in_channels == len(band_idx)+1 and len(band_idx) == weights_in_channels:
    else:
        new_weight[:, :-1] = selected_weight
        new_weight[:, -1] = selected_weight.mean(dim=1)
        # print('added dem weight')

    new_weight = new_weight * (weights_in_channels / new_in_channels)
    # else:
    #     raise Exception("Sorry, channels not match")
    # print(new_weight.shape)

    # print(weights_in_channels / new_in_channels)
    # new_weight = new_weight * (weights_in_channels / new_in_channels)
    module.weight = torch.nn.parameter.Parameter(new_weight)

    return encoder

# ========= 1) 逐尺度特征融合模块 =========
class FeatureFusionPerScale(nn.Module):
    def __init__(self, in_ch_list, out_ch_list):
        """
        in_ch_list: List[int], 每个尺度融合后输入通道数（cat 后的通道）
        out_ch_list: List[int], 解码器期望的每尺度通道数（与 encoder.out_channels 对齐）
        """
        super().__init__()
        assert len(in_ch_list) == len(out_ch_list)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False, padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for in_ch, out_ch in zip(in_ch_list, out_ch_list)
        ])

    def forward(self, feats_cat):
        # feats_cat: List[Tensor]，每个尺度拼接后的特征
        return [blk(f) for blk, f in zip(self.blocks, feats_cat)]

# ========= 2) 简单的卷积式 MLP 头 =========
class ConvMLPHead(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1)
        )
    def forward(self, x):
        return self.head(x)

# ========= 3) 孪生结构主干 =========
class SiameseRTS(pl.LightningModule):
    def __init__(
        self,
        in_bands,                 # Landsat 波段列表（用于权重选择）
        encoder_weights: ResNet50_Weights,
        use_dem: bool,
        out_classes: int = 2,     # 与原模型一致：通道0=mask logits，通道1=heatmap
        encode_lr_mult: float = 0.1,
        loss_cfg: dict = None,
    ):
        super().__init__()
        self.retreat_scale = RETREAT_SCALE
        self.use_dem = use_dem

        # 共享编码器：复用你的方法
        self.shared_encoder = resnet50_encoder(
            weights=encoder_weights,
            in_bands=in_bands,
            use_dem=use_dem,
            depth=5,
            output_stride=32,
        )
        # 基础 Unet++，用于构造解码器与通道配置
        base_unet = UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights=None,   # 我们自己管理 encoder
            in_channels=len(in_bands) + (1 if use_dem else 0),
            classes=out_classes
        )
        # 将 Unet 的 encoder 替换为共享编码器
        base_unet.encoder = self.shared_encoder

        # 两个解码器（结构相同，但参数独立）
        self.slump_decoder = base_unet.decoder
        self.retreat_decoder = copy.deepcopy(base_unet.decoder)

        self.dropout = nn.Dropout(p=0.3)

        # 提前用 dummy 前向推断各尺度通道并构造 fuser（避免训练时 CPU/GPU 混用）
        self.fuser = None
        self._build_fuser_from_dummy(in_ch=len(in_bands) + (1 if use_dem else 0))

        # Encoder 的每尺度通道数（为解码器期望的输入通道）
        # self.enc_out_channels = list(self.shared_encoder.out_channels)  # e.g., [64, 256, 512, 1024, 2048]
        # # 构造逐尺度融合：cat(F_t, F_tminus1, F_t - F_tminus1) → 1x1 conv → enc_out_channels[i]
        # in_ch_list = [2*c + c for c in self.enc_out_channels]  # 3*c
        # out_ch_list = self.enc_out_channels
        # self.fuser = FeatureFusionPerScale(in_ch_list, out_ch_list)

        # 任务头：分割、热力图、头壁后退
        # 解码器输出（高分辨率特征）的通道数可用最后一层 decoder 输出通道来确定
        # SMP Unet++ 的 decoder 输出通道通常等于 decoder_channels[-1]，这里假设为 16（你当前配置）
        dec_out_ch = base_unet.segmentation_head[0].in_channels if isinstance(base_unet.segmentation_head, nn.Sequential) else 16

        # 融合后用于分割与热力图：concat(U_t, U_r)
        self.seg_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64)
        self.heat_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64)
        # 头壁后退：仅用 U_r
        self.retreat_head = ConvMLPHead(in_ch=dec_out_ch, out_ch=1, mid_ch=64) # FOR PREVIOUS VERSIONS ONLY!
        # self.retreat_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64)

        # losses
        self.loss_fn_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn_focal = smp.losses.FocalLoss(smp.losses.BINARY_MODE, alpha=0.5, gamma=2)
        self.loss_fn_mse = nn.MSELoss()
        self.loss_fn_area = AreaPoolLoss(k_size=AREA_K_SIZE, stride=AREA_STRIDE)
        self.loss_fn_awing_weighted = LossWeightedAWing(W=10)
        self.loss_fn_ret = LossWeightedMSE(W=20) # retreat map mse; loss origional=10

        self.loss_weight_dice = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.loss_weight_focal = nn.Parameter(torch.tensor(10.0), requires_grad=False) # 20.0
        self.loss_weight_mse = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.loss_weight_area = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.loss_weight_retreat = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # task-level weights
        self.task_weight_seg = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.task_weight_heat = nn.Parameter(torch.tensor(2.0), requires_grad=False) # 2.0
        self.task_weight_ret = nn.Parameter(torch.tensor(1.0), requires_grad=False) # 2.0

        self.lr = nn.Parameter(torch.tensor(5e-4), requires_grad=False)
        self.encode_lr_mult = nn.Parameter(torch.tensor(encode_lr_mult), requires_grad=False)

        # 缓存每个 epoch 内各 step 的 loss，用于 epoch 结束时打印
        self._epoch_outputs = {"train": [], "valid": [], "test": []}

    # ------------- 构造/兜底 fuser -------------
    def _build_fuser_from_dummy(self, in_ch: int):
        """在 __init__ 使用 dummy encode 推断通道并创建 fuser（注册为子模块，随模型迁移设备）。"""
        H = W = 256  # 保证被 32 整除即可
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            feats_dummy = self.shared_encoder(dummy)  # List[Tensor]
            chs = [f.shape[1] for f in feats_dummy]
            in_ch_list  = [3 * c for c in chs]  # cat(Ft, Ft-1, Ft-Ft-1)
            out_ch_list = chs
            self.fuser = FeatureFusionPerScale(in_ch_list, out_ch_list)

    def _ensure_fuser(self, feats_t: List[torch.Tensor]):
        """极端情况下（通道/层数与 dummy 不同）重建 fuser，并移动到正确设备。"""
        chs = [f.shape[1] for f in feats_t]
        in_ch_list  = [3 * c for c in chs]
        out_ch_list = chs
        need_new = (
            self.fuser is None
            or len(self.fuser.blocks) != len(chs)
            or any(self.fuser.blocks[i][0].in_channels != in_ch_list[i] for i in range(len(chs)))
            or any(self.fuser.blocks[i][0].out_channels != out_ch_list[i] for i in range(len(chs)))
        )
        if need_new:
            fuser = FeatureFusionPerScale(in_ch_list, out_ch_list)
            # 关键：对齐 device/dtype，避免 CUDA/CPU 混用
            fuser.to(device=feats_t[0].device, dtype=feats_t[0].dtype)
            self.fuser = fuser
            
    # ---------------19.01.2026--------------------------
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """安全地传输批次到设备，避免移动冻结的数据类"""
        new_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 只移动张量到设备
                new_batch[key] = value.to(device)
            else:
                # 非张量数据（BoundingBox, str, CRS等）保持原样，不移动到设备
                new_batch[key] = value
        
        return new_batch
    # -----------------------------------------------

    def encode(self, x):
        # 得到多尺度编码器特征列表
        return self.shared_encoder(x)

    def decode_unetpp(self, decoder, feats):
        # 兼容不同版本 SMP：旧版 forward(*feats)，新版 forward(feats)
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)
        try:
            return decoder(*feats)
        except TypeError:
            return decoder(feats)

    def forward(self, image_t, image_tm1, dem_t=None, dem_tm1=None):
        # 拼接 DEM/TPI 通道（标准化 TPI 已在前处理或 batch 中）
        if dem_t is not None and self.use_dem:
            image_t = torch.cat([image_t, dem_t], dim=1)
        if dem_tm1 is not None and self.use_dem:
            image_tm1 = torch.cat([image_tm1, dem_tm1], dim=1)

        # 共享编码器提取特征
        feats_t = self.encode(image_t)
        feats_tm1 = self.encode(image_tm1)

        self._ensure_fuser(feats_t)

        # Slump path：只用 t 年特征
        up_t = self.decode_unetpp(self.slump_decoder, feats_t)

        # Retreat path
        feats_cat = [torch.cat([f_t, f_tm1, (f_t - f_tm1)], dim=1) for f_t, f_tm1 in zip(feats_t, feats_tm1)]
        # fuser 已随模型搬到相同设备；若兜底新建也会被.to 对齐
        fused_feats = self.fuser(feats_cat)
        up_r = self.decode_unetpp(self.retreat_decoder, fused_feats)

        # 上采样特征融合
        merged = torch.cat([up_t, up_r], dim=1)

        if self.training:
            merged = self.dropout(merged)

        # 多任务输出
        logits_mask = self.seg_head(merged)    # [B,1,H,W], from_logits=True
        heatmap = self.heat_head(merged)   # [B,1,H,W], regression
        retreat_map = self.retreat_head(up_r)  # [B,1,H,W], regression # FOR PREVIOUS VERSIONS ONLY!
        # retreat_map = self.retreat_head(merged)  # [B,1,H,W], regression

        return {"logits_mask": logits_mask, "heatmap": heatmap, "retreat": retreat_map}

    # ========== 训练/验证/Test 步骤：与原版一致但扩展到三任务 ==========
    def shared_step(self, batch, stage):
        if self.global_step < 3:
            for name, t in [("heatmap", batch["heatmap"]), ("mask", batch["mask"]), ("retreat_map", batch["retreat_map"])]:
                finite = torch.isfinite(t).all().item()
                print(f"[check] {name} finite={finite}, shape={tuple(t.shape)}, min={float(t.min())}, max={float(t.max())}")
        
        image_t = batch["image_t"]
        image_tm1 = batch["image_tm1"]

        # 检查输入是否包含 NaN/Inf
        if not torch.isfinite(image_t).all():
            print(f"Warning: image_t contains non-finite values")
            return {"loss": torch.tensor(0.0, requires_grad=True, device=image_t.device)}
        
        if not torch.isfinite(image_tm1).all():
            print(f"Warning: image_tm1 contains non-finite values")
            return {"loss": torch.tensor(0.0, requires_grad=True, device=image_tm1.device)}
        
        h, w = image_t.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0

        # 可选 DEM/TPI（标准化 TPI 已在数据管线生成；若训练时还做通道标准化，保持一致）
        dem_t = batch.get("dem_t", None)
        dem_tm1 = batch.get("dem_tm1", None)

        outputs = self.forward(image_t, image_tm1, dem_t, dem_tm1)
        logits_mask = outputs["logits_mask"]
        pred_heat = outputs["heatmap"]
        pred_retreat = outputs["retreat"]

        # 关键：检查模型输出
        if not torch.isfinite(logits_mask).all():
            print(f"Warning: logits_mask contains non-finite values")
            logits_mask = torch.clamp(logits_mask, -10, 10)  # 限制范围
        
        if not torch.isfinite(pred_heat).all():
            print(f"Warning: pred_heat contains non-finite values")
            pred_heat = torch.clamp(pred_heat, 0.0, 2.0)
        
        if not torch.isfinite(pred_retreat).all():
            print(f"Warning: pred_retreat contains non-finite values")
            pred_retreat = torch.clamp(pred_retreat, 0.0, 100.0)

        # def safe_mean(x):
        #     return torch.nanmean(torch.where(torch.isfinite(x), x, torch.zeros_like(x)))
        # print("means:", {
        # "dice": float(safe_mean(loss_dice)),
        # "focal": float(safe_mean(loss_focal)),
        # "heat": float(safe_mean(loss_heat)),
        # "ret": float(safe_mean(loss_ret)),
        # "area": float(safe_mean(loss_area)),
        # })

        # 目标
        # gt_mask = batch["mask"].unsqueeze(1)              # [B,1,H,W] 二元
        # gt_heatmap = batch["heatmap"].unsqueeze(1)           # [B,1,H,W] 连续
        # gt_retreat = batch["retreat_map"].unsqueeze(1)       # [B,1,H,W] 连续（你的“头壁后退图”）

        def ensure_ch1(t: torch.Tensor) -> torch.Tensor:
            # 标准化为 [B,1,H,W]，仅当标签为 [B,H,W] 时才 unsqueeze
            return t.unsqueeze(1) if t.dim() == 3 else t

        gt_mask = ensure_ch1(batch["mask"])         # [B,1,H,W]
        gt_heatmap = ensure_ch1(batch["heatmap"])      # [B,1,H,W]
        mask_heat = torch.isfinite(gt_heatmap)  # [B,1,H,W]
        gt_retreat = ensure_ch1(batch["retreat_map"])  # [B,1,H,W]

        # valid_retreat = torch.isfinite(gt_retreat)

        # 断言确保与输出形状一致
        assert gt_mask.shape == logits_mask.shape, f"mask shape {gt_mask.shape} != {logits_mask.shape}"
        assert gt_heatmap.shape == pred_heat.shape, f"heat shape {gt_heatmap.shape} != {pred_heat.shape}"
        assert gt_retreat.shape == pred_retreat.shape, f"retreat shape {gt_retreat.shape} != {pred_retreat.shape}"

        # 分割损失
        loss_dice = self.loss_fn_dice(logits_mask, gt_mask)
        loss_focal = self.loss_fn_focal(logits_mask, gt_mask)

        # 热力图与后退图损失
        loss_heat = self.loss_fn_awing_weighted(
            pred_heat[mask_heat], 
            gt_heatmap[mask_heat], 
            batch["mask"][mask_heat])
        loss_area  = self.loss_fn_area(
            torch.where(mask_heat, pred_heat, torch.zeros_like(pred_heat)),
            torch.where(mask_heat, gt_heatmap, torch.zeros_like(gt_heatmap)))

        # valid_pix = torch.isfinite(gt_retreat)  # [B,1,H,W]
        # per_pix_mse = (pred_retreat - gt_retreat) ** 2                # [B,1,H,W]
        # per_pix_mse = torch.where(valid_pix, per_pix_mse, torch.zeros_like(per_pix_mse))

        # # 跳过“全零标签样本”
        # sample_has_retreat = (gt_retreat.abs().sum(dim=(1,2,3)) > 0).float()  # [B]
        # if sample_has_retreat.sum() > 0:
        #     # 按样本平均（只对有 retreat 的样本）
        #     # 也可改为“按像元平均”（micro），看你的评估偏好
        #     per_sample = per_pix_mse.mean(dim=(1,2,3))  # [B]
        #     loss_ret = (per_sample * sample_has_retreat).sum() / sample_has_retreat.sum()
        # else:
        #     loss_ret = torch.tensor(0., device=gt_retreat.device)

        # ---------- Retreat loss：使用 LossWeightedMSE ----------
        # 目标: pred_retreat, gt_retreat 形状 [B,1,H,W]，范围 0~1（已除以 RETREAT_SCALE）

        # 1) 处理无效像素（NaN/Inf）
        valid_pix = torch.isfinite(gt_retreat)
        pred_ret_valid = torch.where(valid_pix, pred_retreat, torch.zeros_like(pred_retreat))
        gt_ret_valid   = torch.where(valid_pix, gt_retreat,   torch.zeros_like(gt_retreat))

        # 2) 构造 weight_mask：用分割 mask 作为权重区域
        #    mask 为 [B,1,H,W]，0/1，表示崩塌区域；崩塌区域内权重大
        weight_mask = ensure_ch1(batch["mask"]).to(gt_ret_valid.device).float()  # [B,1,H,W]

        # 3) 只在“有 retreat 的样本”上计算 loss
        sample_has_retreat = (gt_ret_valid.abs().sum(dim=(1,2,3)) > 0)  # [B] bool
        if sample_has_retreat.any():
            # 只取有 retreat 的样本进行加权 MSE
            pred_ret_sel = pred_ret_valid[sample_has_retreat]   # [B_sel,1,H,W]
            gt_ret_sel = gt_ret_valid[sample_has_retreat]     # [B_sel,1,H,W]
            mask_sel = weight_mask[sample_has_retreat]      # [B_sel,1,H,W]

            # LossWeightedMSE: MSE((output-target)^2 * (1 + W*mask))
        #     loss_ret = self.loss_fn_ret(pred_ret_sel, gt_ret_sel, mask_sel)
        # else:
        #     loss_ret = torch.tensor(0.0, device=gt_retreat.device)
            loss_ret_fine = self.loss_fn_ret(pred_ret_sel, gt_ret_sel, mask_sel)

            # ---------- Retreat coarse-scale 一致性约束 ----------
            # 在较大 kernel 上对齐 retreat 形状，减少只学局部纹理的情况
            pool = torch.nn.AvgPool2d(kernel_size=8, stride=8)  # 256x256 -> 32x32
            pred_coarse = pool(pred_ret_sel)
            gt_coarse   = pool(gt_ret_sel)
            loss_ret_coarse = F.mse_loss(pred_coarse, gt_coarse)

            lambda_coarse = 0.3  # 可调，先试 0.5
            loss_ret = loss_ret_fine + lambda_coarse * loss_ret_coarse
        else:
            loss_ret = torch.tensor(0.0, device=gt_retreat.device)

        # 4) 应用 MSE 任务权重
        loss_ret = self.loss_weight_mse * loss_ret
        # -------------------------------------------------------

        # loss_ret = self.loss_fn_mse(pred_retreat, gt_retreat)  # 或使用 awing/masked MSE

        # 1) 各子任务内部先组合
        loss_seg  = (self.loss_weight_dice  * loss_dice +
                    self.loss_weight_focal * loss_focal)
        loss_heat_total = (self.loss_weight_mse  * loss_heat +
                        self.loss_weight_area * loss_area)
        loss_ret_total  = self.loss_weight_retreat * loss_ret  # 或者直接用 loss_ret，看你是否想共用 mse 权重

        # 2) 任务级别权重（初始三个都是 1.0）
        loss = (self.task_weight_seg * loss_seg +
                self.task_weight_heat * loss_heat_total +
                self.task_weight_ret * loss_ret_total)

        # 计算指标（IoU）
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt_mask.long(), mode="binary")

        if self.global_step < 3:
            def isfin(x): return torch.isfinite(x).item() if x.numel()==1 else torch.isfinite(x).all().item()
            print("loss finite:", {
                "dice":   isfin(loss_dice),
                "focal":  isfin(loss_focal),
                "heat":   isfin(loss_heat),
                "ret":    isfin(loss_ret),
                "area":   isfin(loss_area),
            })

        return {
            "loss": loss,
            "loss_dice": loss_dice, "loss_focal": loss_focal,
            "loss_heat": loss_heat, "loss_ret": loss_ret, "loss_area": loss_area,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "outputs": outputs,
        }

    def shared_epoch_end(self, outputs, stage):
        # 聚合与日志（与原来类似）
        loss = torch.mean(torch.stack([x["loss"] for x in outputs]), dim=0)
        loss_dice  = torch.mean(torch.stack([x["loss_dice"] for x in outputs]), dim=0)
        loss_focal = torch.mean(torch.stack([x["loss_focal"] for x in outputs]), dim=0)
        loss_heat = torch.mean(torch.stack([x["loss_heat"] for x in outputs]), dim=0)
        loss_ret = torch.mean(torch.stack([x["loss_ret"] for x in outputs]), dim=0)
        loss_area = torch.mean(torch.stack([x["loss_area"] for x in outputs]), dim=0)
        tp = torch.cat([x["tp"] for x in outputs]); fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs]); tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_dice": loss_dice,
            f"{stage}_loss_focal": loss_focal,
            f"{stage}_loss_heat": loss_heat,
            f"{stage}_loss_retreat": loss_ret,
            f"{stage}_loss_aresa": loss_area,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
        }
        self.log_dict(metrics, prog_bar=True)

        print(f"[{stage} epoch end] "
              f"loss={float(loss):.4f}, "
              f"dice={float(loss_dice):.4f}, "
              f"focal={float(loss_focal):.4f}, "
              f"heat={float(loss_heat):.4f}, "
              f"ret={float(loss_ret):.4f}, "
              f"area={float(loss_area):.4f}")

    def training_step(self, batch, batch_idx): 
        info = self.shared_step(batch, "train")

        # 记录到缓存，供 on_train_epoch_end 聚合
        self._epoch_outputs["train"].append({
            "loss": info["loss"].detach(),
            "loss_dice": info["loss_dice"].detach(),
            "loss_focal": info["loss_focal"].detach(),
            "loss_heat": info["loss_heat"].detach(),
            "loss_ret": info["loss_ret"].detach(),
            "loss_area": info["loss_area"].detach(),
        })

        # 在 epoch 级别聚合
        self.log("train_loss", info["loss"], on_step=False, on_epoch=True, prog_bar=True)
        # 如果要看 dice/focal 等，也可以：
        self.log("train_loss_dice",   info["loss_dice"],  on_step=False, on_epoch=True)
        self.log("train_loss_focal",  info["loss_focal"], on_step=False, on_epoch=True)
        self.log("train_loss_heat",   info["loss_heat"],  on_step=False, on_epoch=True)
        self.log("train_loss_retreat",info["loss_ret"],   on_step=False, on_epoch=True)
        self.log("train_loss_area",   info["loss_area"],  on_step=False, on_epoch=True)
        return info

    def validation_step(self, batch, batch_idx):
        info = self.shared_step(batch, "valid")

        self._epoch_outputs["valid"].append({
            "loss": info["loss"].detach(),
            "loss_dice": info["loss_dice"].detach(),
            "loss_focal": info["loss_focal"].detach(),
            "loss_heat": info["loss_heat"].detach(),
            "loss_ret": info["loss_ret"].detach(),
            "loss_area": info["loss_area"].detach(),
        })

        # 关键：在验证阶段写 valid_loss（EarlyStopping/ModelCheckpoint 监控这个）
        self.log("valid_loss", info["loss"], on_step=False, on_epoch=True, prog_bar=True)
        # 可选更多：
        self.log("valid_loss_dice", info["loss_dice"],  on_step=False, on_epoch=True)
        self.log("valid_loss_focal", info["loss_focal"], on_step=False, on_epoch=True)
        self.log("valid_loss_heat", info["loss_heat"],  on_step=False, on_epoch=True)
        self.log("valid_loss_retreat", info["loss_ret"],   on_step=False, on_epoch=True)
        self.log("valid_loss_area", info["loss_area"],  on_step=False, on_epoch=True)
        return info

    def test_step(self, batch, batch_idx):
        info = self.shared_step(batch, "test")

        self._epoch_outputs["test"].append({
            "loss": info["loss"].detach(),
            "loss_dice": info["loss_dice"].detach(),
            "loss_focal": info["loss_focal"].detach(),
            "loss_heat": info["loss_heat"].detach(),
            "loss_ret": info["loss_ret"].detach(),
            "loss_area": info["loss_area"].detach(),
        })

        self.log("test_loss", info["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return info
    
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        双时相预测：
        输入 batch: {'image_t','image_tm1','dem_t','dem_tm1',...}
        输出: dict，包含年 t/t-1 的影像和预测的 retreat/heatmap/mask 概率。
        """

        image_t = batch["image_t"]
        image_tm1 = batch["image_tm1"]
        dem_t = batch.get("dem_t", None)
        dem_tm1 = batch.get("dem_tm1", None)
        
        # DEBUG
        # print(f"Input shapes: image_t={image_t.shape}, image_tm1={image_tm1.shape}")

        original_size = image_t.shape[-2:]
        h, w = original_size

        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            # 使用padding而不是resize，减少插值伪影
            image_t = F.pad(image_t, (0, pad_w, 0, pad_h), mode='reflect')
            image_tm1 = F.pad(image_tm1, (0, pad_w, 0, pad_h), mode='reflect')
            
            if dem_t is not None:
                dem_t = F.pad(dem_t, (0, pad_w, 0, pad_h), mode='reflect')
            if dem_tm1 is not None:
                dem_tm1 = F.pad(dem_tm1, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            outputs = self.forward(image_t, image_tm1, dem_t, dem_tm1)
            logits_mask = outputs["logits_mask"]

            if pad_h > 0 or pad_w > 0:
                logits_mask = logits_mask[:, :, :h, :w]
                outputs["heatmap"] = outputs["heatmap"][:, :, :h, :w]
                outputs["retreat"] = outputs["retreat"][:, :, :h, :w]
                
            # pred_heat = outputs["heatmap"]
            heatmap = outputs["heatmap"]
            # pred_retreat = outputs["retreat"] * self.retreat_scale  # 恢复真实尺度
            retreat = outputs["retreat"] * self.retreat_scale  # 恢复真实尺度
            prob_mask = logits_mask.sigmoid()

        return {
            "image_t": image_t.detach().cpu(),
            "image_tm1": image_tm1.detach().cpu(),
            # "pred_retreat": pred_retreat.detach().cpu(),
            # "pred_heat": pred_heat.detach().cpu(),
            "retreat":   retreat.detach().cpu(),
            "heatmap": heatmap.detach().cpu(),
            "prob_mask": prob_mask.detach().cpu(),
        }

        # --------- 辅助函数：聚合并打印 ---------
    def _aggregate_and_print_epoch(self, stage: str):
        outputs = self._epoch_outputs[stage]
        if len(outputs) == 0:
            return

        device = outputs[0]["loss"].device
        def stack_and_mean(key):
            return torch.mean(torch.stack([x[key].to(device) for x in outputs]), dim=0)

        loss       = stack_and_mean("loss")
        loss_dice  = stack_and_mean("loss_dice")
        loss_focal = stack_and_mean("loss_focal")
        loss_heat  = stack_and_mean("loss_heat")
        loss_ret   = stack_and_mean("loss_ret")
        loss_area  = stack_and_mean("loss_area")

        # 终端打印
        print(
            f"[{stage} epoch end] "
            f"loss={float(loss):.4f}, "
            f"dice={float(loss_dice):.4f}, "
            f"focal={float(loss_focal):.4f}, "
            f"heat={float(loss_heat):.4f}, "
            f"ret={float(loss_ret):.4f}, "
            f"area={float(loss_area):.4f}"
        )

        # 清空缓存，准备下一个 epoch
        self._epoch_outputs[stage] = []

    # --------- Lightning v2 推荐的 epoch hooks ---------
    def on_train_epoch_end(self):
        self._aggregate_and_print_epoch("train")

    def on_validation_epoch_end(self):
        self._aggregate_and_print_epoch("valid")

    def on_test_epoch_end(self):
        self._aggregate_and_print_epoch("test")

    def configure_optimizers(self):
        params = list(self.named_parameters())
        def is_backbone(n): return "shared_encoder" in n
        grouped_parameters = [
            {"params": [p for n,p in params if is_backbone(n)], "lr": float(self.lr)*float(self.encode_lr_mult)},
            {"params": [p for n,p in params if not is_backbone(n)], "lr": float(self.lr)},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=float(self.lr), weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # 每 10 个 epoch 重启
        T_mult=1,
        eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            }
        }