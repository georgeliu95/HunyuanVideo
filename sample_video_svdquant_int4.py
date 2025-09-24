import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import torch
import torch.nn as nn

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.
    Args:
        x: the dividend.
        y: the divisor.
    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def fake_quant_per_channel(x):
    # 逐通道计算scale (卷积权重的形状: [out_c, in_c, kH, kW])
    max_vals = torch.amax(torch.abs(x.detach()), dim=(-1), keepdim=True)
    scales = max_vals / 7.0
    scales = torch.clamp(scales, min=1e-8)
    
    # 量化 & 反量化
    q_x = torch.clamp(torch.round(x / scales), -7, 7)
    x_dequant = q_x * scales
    
    return x_dequant

def fake_svdquant_weight(weight, rank=64):
    dtype = weight.dtype
    device = weight.device
    #ori_weight = weight
    #u, s, vh = torch.linalg.svd(weight.cpu().float())
    u, s, vh = torch.linalg.svd(weight.float())
    us = u[:, : rank] * s[: rank]
    vh = vh[: rank]
    lora = torch.mm(us, vh)
    #residual_ori = weight.cpu().float() - lora
    residual_ori = weight.float() - lora
    residual = fake_quant_per_channel(residual_ori)
    quant_residual = residual_ori - residual
    quant_w = torch.mm(us, vh)
    quant = torch.norm(quant_residual) < 1000
    quant = quant.item()
    if quant:
        return True, us.to(dtype).to(device), vh.to(dtype).to(device), residual.to(dtype).to(device)
    else:
        return False, None, None, None

def int4_quantize_deepgemm_with_padding(x, int4_max=7.0, eps=1e-3):
    dtype = x.dtype
    ndim = x.ndim
    if ndim == 3:
        x = x.squeeze(0)
    m, n = x.shape
    x_padded = torch.zeros((m, ceil_div(n, 256) * 256), dtype=x.dtype, device=x.device)
    x_padded[:m,:n] = x
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(eps).unsqueeze(2)
    x_padded_view = torch.clamp(torch.round(x_padded_view * (int4_max / x_amax)), -7, 7)
    x_dequant = (x_padded_view * (x_amax / int4_max)).view(m, -1)
    x_dequant = x_dequant[:m, :n]
    if ndim == 3:
        x_dequant = x_dequant.unsqueeze(0)
    return x_dequant.to(dtype)
    #return x_padded_view, (x_amax / int4_max).view(m, -1)

class INT4Linear_svdquant(nn.Module):
    def __init__(self, ori_linear: nn.Linear):
        super().__init__()
        assert isinstance(ori_linear, nn.Linear)
        self.in_features = ori_linear.in_features
        self.out_features = ori_linear.out_features
        self.bias = ori_linear.bias

        self.n = ori_linear.weight.shape[0]
        self.k = ori_linear.weight.shape[1]

        ### svdquant weight
        use_quant, us, vh, residual = fake_svdquant_weight(ori_linear.weight)
        self.use_quant = use_quant
        if self.use_quant:
            self.us = nn.Parameter(us, requires_grad=False)
            self.vh = nn.Parameter(vh, requires_grad=False)
            self.residual = nn.Parameter(residual, requires_grad=False)
            del ori_linear.weight
        else:
            self.weight = ori_linear.weight
        print('convert int4 weight, use_quant: ', self.use_quant)

    def forward(self, x):
        if self.use_quant:
            weight = torch.mm(self.us, self.vh)
            l_out = torch.nn.functional.linear(x, weight, self.bias)
            fq_x = int4_quantize_deepgemm_with_padding(x)
            r_out = torch.nn.functional.linear(fq_x, self.residual)
            l_out = l_out + r_out
        else:
            l_out = torch.nn.functional.linear(x, self.weight, self.bias)
        return l_out

def set_int4_layer(model):
    for block in model.single_blocks:
        for name, module in block.named_children():
            if name == "linear1" or name == "linear2":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)
    for block in model.double_blocks:
        for name, module in block.named_children():
            if name == "img_mlp":
                for subname, submodule in module.named_children():
                    if subname == "fc1" or subname == "fc2":
                        wrapped_submodule = INT4Linear_svdquant(submodule)
                        setattr(module, subname, wrapped_submodule)
            elif name == "img_attn_qkv":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)
            elif name == "img_attn_proj":
                wrapped_module = INT4Linear_svdquant(module)
                setattr(block, name, wrapped_module)

def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    set_int4_layer(hunyuan_video_sampler.pipeline.transformer)
    #import sys; sys.exit(0)
    #import pdb; pdb.set_trace()
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
