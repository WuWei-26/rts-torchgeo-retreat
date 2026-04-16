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
class FeatureFusionPerScale(nn.Module):
    def __init__(self, in_ch_list, out_ch_list):
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
        # feats_cat: List[Tensor]
        return [blk(f) for blk, f in zip(self.blocks, feats_cat)]

# ConvMLP
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
class SiameseRTS(pl.LightningModule):
    def __init__(
        self,
        in_bands,
        encoder_weights: ResNet50_Weights,
        use_dem: bool,
        out_classes: int = 2,
        encode_lr_mult: float = 0.1,
        loss_cfg: dict = None,
    ):
        super().__init__()
        self.retreat_scale = RETREAT_SCALE
        self.use_dem = use_dem

        self.shared_encoder = resnet50_encoder(
            weights=encoder_weights,
            in_bands=in_bands,
            use_dem=use_dem,
            depth=5,
            output_stride=32,
        )

        base_unet = UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=len(in_bands) + (1 if use_dem else 0),
            classes=out_classes
        )

        base_unet.encoder = self.shared_encoder
        self.slump_decoder = base_unet.decoder
        self.retreat_decoder = copy.deepcopy(base_unet.decoder)
        self.dropout = nn.Dropout(p=0.3)
        self.fuser = None
        self._build_fuser_from_dummy(in_ch=len(in_bands) + (1 if use_dem else 0))

        dec_out_ch = base_unet.segmentation_head[0].in_channels if isinstance(base_unet.segmentation_head, nn.Sequential) else 16

        self.seg_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64) #2*dec_out_ch
        self.heat_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64) #2*dec_out_ch
        self.retreat_head = ConvMLPHead(in_ch=2*dec_out_ch, out_ch=1, mid_ch=64) #2*dec_out_ch
        # losses
        self.loss_fn_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn_focal = smp.losses.FocalLoss(smp.losses.BINARY_MODE, alpha=0.5, gamma=2)
        self.loss_fn_mse = nn.MSELoss()
        self.loss_fn_area = AreaPoolLoss(k_size=AREA_K_SIZE, stride=AREA_STRIDE)
        self.loss_fn_awing_weighted = LossWeightedAWing(W=10)
        self.loss_fn_ret = LossWeightedMSE(W=20) # retreat map mse; loss origional=10

        self.loss_weight_dice = nn.Parameter(torch.tensor(1.0), requires_grad=False) # 1.0
        self.loss_weight_focal = nn.Parameter(torch.tensor(1.0), requires_grad=False) # 20.0
        self.loss_weight_mse = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.loss_weight_area = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.loss_weight_retreat = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # task-level weights
        self.task_weight_seg = nn.Parameter(torch.tensor(1.0), requires_grad=False) # 1.0
        self.task_weight_heat = nn.Parameter(torch.tensor(2.0), requires_grad=False) # 2.0
        self.task_weight_ret = nn.Parameter(torch.tensor(1.0), requires_grad=False) # 1.0

        self.lr = nn.Parameter(torch.tensor(1e-4), requires_grad=False) # 1e-4 for training, 1e-5 for fine tuning
        self.encode_lr_mult = nn.Parameter(torch.tensor(encode_lr_mult), requires_grad=False)

        self._epoch_outputs = {"train": [], "valid": [], "test": []}

    def _build_fuser_from_dummy(self, in_ch: int):
        H = W = 256
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            feats_dummy = self.shared_encoder(dummy)  # List[Tensor]
            chs = [f.shape[1] for f in feats_dummy]
            in_ch_list  = [3 * c for c in chs]  # cat(Ft, Ft-1, Ft-Ft-1)
            out_ch_list = chs
            self.fuser = FeatureFusionPerScale(in_ch_list, out_ch_list)

    def _ensure_fuser(self, feats_t: List[torch.Tensor]):
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
            fuser.to(device=feats_t[0].device, dtype=feats_t[0].dtype)
            self.fuser = fuser
            
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        new_batch = {} 
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                new_batch[key] = value.to(device)
            else:
                new_batch[key] = value
        
        return new_batch

    def encode(self, x):
        return self.shared_encoder(x)

    def decode_unetpp(self, decoder, feats):
        if not isinstance(feats, (list, tuple)):
            feats = list(feats)
        try:
            return decoder(*feats)
        except TypeError:
            return decoder(feats)

    def forward(self, image_t, image_tm1, dem_t=None, dem_tm1=None):
        if dem_t is not None and self.use_dem:
            image_t = torch.cat([image_t, dem_t], dim=1)
        if dem_tm1 is not None and self.use_dem:
            image_tm1 = torch.cat([image_tm1, dem_tm1], dim=1)

        feats_t = self.encode(image_t)
        feats_tm1 = self.encode(image_tm1)
        self._ensure_fuser(feats_t)

        feats_cat = [torch.cat([f_t, f_tm1, (f_t - f_tm1)], dim=1) 
                     for f_t, f_tm1 in zip(feats_t, feats_tm1)]
        fused_feats = self.fuser(feats_cat)

        # seg + heatmap path
        # up = self.decode_unetpp(self.slump_decoder, fused_feats)
        # if self.training:
        #     up = self.dropout(up)
        # logits_mask = self.seg_head(up)
        # heatmap     = self.heat_head(up)

        # retreat path, isolated gradient
        # fused_for_retreat = [f.detach() for f in fused_feats] \
        #             if not self.training else fused_feats
        # up_r = self.decode_unetpp(self.retreat_decoder, fused_for_retreat)
        # retreat_map = self.retreat_head(up_r)

        up_t = self.decode_unetpp(self.slump_decoder, feats_t) # learn year t features
        up_r = self.decode_unetpp(self.retreat_decoder, fused_feats) # learn difference
        merged = torch.cat([up_t, up_r], dim=1)

        if self.training:
            merged = self.dropout(merged)
        
        logits_mask=self.seg_head(merged)
        heatmap=self.heat_head(merged)
        # retreat_map = self.retreat_head(up_r)  # [B,1,H,W], regression # FOR PREVIOUS VERSIONS ONLY
        retreat_map = self.retreat_head(merged)  # [B,1,H,W], regression

        return {"logits_mask": logits_mask, "heatmap": heatmap, "retreat": retreat_map}

    def shared_step(self, batch, stage):
        if self.global_step < 3:
            for name, t in [("heatmap", batch["heatmap"]), ("mask", batch["mask"]), ("retreat_map", batch["retreat_map"])]:
                finite = torch.isfinite(t).all().item()
                print(f"[check] {name} finite={finite}, shape={tuple(t.shape)}, min={float(t.min())}, max={float(t.max())}")
        
        image_t = batch["image_t"]
        image_tm1 = batch["image_tm1"]

        # NaN/Inf check
        if not torch.isfinite(image_t).all():
            print(f"Warning: image_t contains non-finite values")
            return {"loss": torch.tensor(0.0, requires_grad=True, device=image_t.device)}
        
        if not torch.isfinite(image_tm1).all():
            print(f"Warning: image_tm1 contains non-finite values")
            return {"loss": torch.tensor(0.0, requires_grad=True, device=image_tm1.device)}
        
        h, w = image_t.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0

        dem_t = batch.get("dem_t", None)
        dem_tm1 = batch.get("dem_tm1", None)

        outputs = self.forward(image_t, image_tm1, dem_t, dem_tm1)
        logits_mask = outputs["logits_mask"]
        pred_heat = outputs["heatmap"]
        pred_retreat = outputs["retreat"]

        if not torch.isfinite(logits_mask).all():
            print(f"Warning: logits_mask contains non-finite values")
            logits_mask = torch.clamp(logits_mask, -10, 10)
        
        if not torch.isfinite(pred_heat).all():
            print(f"Warning: pred_heat contains non-finite values")
            pred_heat = torch.clamp(pred_heat, 0.0, 2.0)
        
        if not torch.isfinite(pred_retreat).all():
            print(f"Warning: pred_retreat contains non-finite values")
            pred_retreat = torch.clamp(pred_retreat, 0.0, 100.0) # retreat > 0

        def ensure_ch1(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1) if t.dim() == 3 else t

        gt_mask = ensure_ch1(batch["mask"])
        gt_heatmap = ensure_ch1(batch["heatmap"])
        mask_heat = torch.isfinite(gt_heatmap)
        gt_retreat = ensure_ch1(batch["retreat_map"])

        assert gt_mask.shape == logits_mask.shape, f"mask shape {gt_mask.shape} != {logits_mask.shape}"
        assert gt_heatmap.shape == pred_heat.shape, f"heat shape {gt_heatmap.shape} != {pred_heat.shape}"
        assert gt_retreat.shape == pred_retreat.shape, f"retreat shape {gt_retreat.shape} != {pred_retreat.shape}"

        loss_dice = self.loss_fn_dice(logits_mask, gt_mask)
        loss_focal = self.loss_fn_focal(logits_mask, gt_mask)

        loss_heat = self.loss_fn_awing_weighted(
            pred_heat[mask_heat], 
            gt_heatmap[mask_heat], 
            batch["mask"][mask_heat])

        loss_area  = self.loss_fn_area(
            torch.where(mask_heat, pred_heat, torch.zeros_like(pred_heat)),
            torch.where(mask_heat, gt_heatmap, torch.zeros_like(gt_heatmap)))

        #（NaN/Inf）
        valid_pix = torch.isfinite(gt_retreat)
        pred_ret_valid = torch.where(valid_pix, pred_retreat, torch.zeros_like(pred_retreat))
        gt_ret_valid = torch.where(valid_pix, gt_retreat, torch.zeros_like(gt_retreat))

        weight_mask = ensure_ch1(batch["mask"]).to(gt_ret_valid.device).float()  # [B,1,H,W]

        sample_has_retreat = (gt_ret_valid.abs().sum(dim=(1,2,3)) > 0)  # [B] bool
        if sample_has_retreat.any():
            pred_ret_sel = pred_ret_valid[sample_has_retreat]
            gt_ret_sel = gt_ret_valid[sample_has_retreat]
            mask_sel = weight_mask[sample_has_retreat]

            # LossWeightedMSE: MSE((output-target)^2 * (1 + W*mask))
        #     loss_ret = self.loss_fn_ret(pred_ret_sel, gt_ret_sel, mask_sel)
        # else:
        #     loss_ret = torch.tensor(0.0, device=gt_retreat.device)
            loss_ret_fine = self.loss_fn_ret(pred_ret_sel, gt_ret_sel, mask_sel)

            pool = torch.nn.AvgPool2d(kernel_size=8, stride=8)  # 256x256 -> 32x32
            pred_coarse = pool(pred_ret_sel)
            gt_coarse   = pool(gt_ret_sel)
            loss_ret_coarse = F.mse_loss(pred_coarse, gt_coarse)

            lambda_coarse = 0.3
            loss_ret = loss_ret_fine + lambda_coarse * loss_ret_coarse
        else:
            loss_ret = torch.tensor(0.0, device=gt_retreat.device)

        loss_ret = self.loss_weight_mse * loss_ret
        # loss_ret = self.loss_fn_mse(pred_retreat, gt_retreat)E

        loss_seg = (self.loss_weight_dice*loss_dice+self.loss_weight_focal*loss_focal)
        loss_heat_total = (self.loss_weight_mse*loss_heat+self.loss_weight_area*loss_area)
        loss_ret_total  = self.loss_weight_retreat * loss_ret

        loss = (self.task_weight_seg * loss_seg +
                self.task_weight_heat * loss_heat_total +
                self.task_weight_ret * loss_ret_total)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt_mask.long(), mode="binary")

        if self.global_step < 3:
            def isfin(x): return torch.isfinite(x).item() if x.numel()==1 else torch.isfinite(x).all().item()
            print("loss finite:", {
                "dice": isfin(loss_dice),
                "focal": isfin(loss_focal),
                "heat": isfin(loss_heat),
                "ret": isfin(loss_ret),
                "area": isfin(loss_area),
            })

        return {
            "loss": loss,
            "loss_dice": loss_dice, "loss_focal": loss_focal,
            "loss_heat": loss_heat, "loss_ret": loss_ret, "loss_area": loss_area,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "outputs": outputs,
        }

    def shared_epoch_end(self, outputs, stage):
        loss = torch.mean(torch.stack([x["loss"] for x in outputs]), dim=0)
        loss_dice = torch.mean(torch.stack([x["loss_dice"] for x in outputs]), dim=0)
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
        if batch is None:
            return None
        
        info = self.shared_step(batch, "train")

        self._epoch_outputs["train"].append({
            "loss": info["loss"].detach(),
            "loss_dice": info["loss_dice"].detach(),
            "loss_focal": info["loss_focal"].detach(),
            "loss_heat": info["loss_heat"].detach(),
            "loss_ret": info["loss_ret"].detach(),
            "loss_area": info["loss_area"].detach(),
        })

        self.log("train_loss", info["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_dice", info["loss_dice"], on_step=False, on_epoch=True)
        self.log("train_loss_focal", info["loss_focal"], on_step=False, on_epoch=True)
        self.log("train_loss_heat", info["loss_heat"], on_step=False, on_epoch=True)
        self.log("train_loss_retreat", info["loss_ret"], on_step=False, on_epoch=True)
        self.log("train_loss_area", info["loss_area"], on_step=False, on_epoch=True)
        return info

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        
        info = self.shared_step(batch, "valid")

        self._epoch_outputs["valid"].append({
            "loss": info["loss"].detach(),
            "loss_dice": info["loss_dice"].detach(),
            "loss_focal": info["loss_focal"].detach(),
            "loss_heat": info["loss_heat"].detach(),
            "loss_ret": info["loss_ret"].detach(),
            "loss_area": info["loss_area"].detach(),
        })

        self.log("valid_loss", info["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_loss_dice", info["loss_dice"], on_step=False, on_epoch=True)
        self.log("valid_loss_focal", info["loss_focal"], on_step=False, on_epoch=True)
        self.log("valid_loss_heat", info["loss_heat"], on_step=False, on_epoch=True)
        self.log("valid_loss_retreat", info["loss_ret"], on_step=False, on_epoch=True)
        self.log("valid_loss_area", info["loss_area"], on_step=False, on_epoch=True)
        return info

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
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
            # pred_retreat = outputs["retreat"] * self.retreat_scale
            retreat = outputs["retreat"] * self.retreat_scale
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

    def _aggregate_and_print_epoch(self, stage: str):
        # outputs = self._epoch_outputs[stage]
        outputs = [x for x in self._epoch_outputs[stage] if x is not None]
        if len(outputs) == 0:
            return

        device = outputs[0]["loss"].device
        def stack_and_mean(key):
            return torch.mean(torch.stack([x[key].to(device) for x in outputs]), dim=0)

        loss = stack_and_mean("loss")
        loss_dice = stack_and_mean("loss_dice")
        loss_focal = stack_and_mean("loss_focal")
        loss_heat = stack_and_mean("loss_heat")
        loss_ret = stack_and_mean("loss_ret")
        loss_area = stack_and_mean("loss_area")

        print(
            f"[{stage} epoch end] "
            f"loss={float(loss):.4f}, "
            f"dice={float(loss_dice):.4f}, "
            f"focal={float(loss_focal):.4f}, "
            f"heat={float(loss_heat):.4f}, "
            f"ret={float(loss_ret):.4f}, "
            f"area={float(loss_area):.4f}"
        )

        self._epoch_outputs[stage] = []

    def on_train_epoch_end(self):
        self._epoch_outputs["train"] = [
        x for x in self._epoch_outputs["train"] if x is not None
    ]
        self._aggregate_and_print_epoch("train")

    def on_validation_epoch_end(self):
        self._epoch_outputs["valid"] = [
        x for x in self._epoch_outputs["valid"] if x is not None
    ]
        self._aggregate_and_print_epoch("valid")

    def on_test_epoch_end(self):
        self._epoch_outputs["test"] = [
        x for x in self._epoch_outputs["test"] if x is not None
    ]
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
        T_0=10,
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