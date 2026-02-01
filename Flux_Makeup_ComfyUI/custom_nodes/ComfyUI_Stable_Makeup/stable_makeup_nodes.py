# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import folder_paths
import os
import torch
import pdb
import cv2

from diffusers import (DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
                       DDPMScheduler, LCMScheduler, StableDiffusionPipeline, )
from .pipeline_sd15 import StableDiffusionControlNetPipeline
from .detail_encoder.encoder_plus import detail_encoder
from .import spiga_draw
from PIL import Image, ImageFilter
from .facelib import FaceDetector
from comfy.utils import common_upscale
import numpy as np
from torchvision import transforms
from .mask_model import BiSeNet
import copy

# 新增：为集成 eval.py 中的生成逻辑做准备（按需导入）
import sys
# project-2 src 目录（根据你的工程结构硬编码为绝对路径）
_src = "custom_nodes/ComfyUI_Stable_Makeup"
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

# 尝试导入 eval.py 使用的类，如果不存在则安全回退（不影响原有 pipeline）
try:
    from src.pipeline import FluxPipeline
    from src.transformer_flux import FluxTransformer2DModel
    from src.attn_processor import MakeupFluxAttnProcessor, ImageProjModel
    from torchvision import transforms as _tv_transforms
    from einops import rearrange, repeat
    _HAS_FLUX = True
    print("Successfully imported FluxPipeline and related classes!!!!!!!!!!!!!!!!!!!!!!!!!!!")
except Exception:
    _HAS_FLUX = False

class WrappedPipe:
    def __init__(self, pipe, ckpt_path=None):
        super().__init__()
        self.device = pipe.device
        self.pipe = pipe

        # 加载 ref_image_proj_model
        self.ref_image_proj_model = ImageProjModel().to(device=self.device, dtype=torch.bfloat16)
        state_dict = torch.load(ckpt_path)

        # 加载 transformer 权重
        pipe_transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in state_dict.items() if k.startswith("transformer.")
        }
        self.pipe.transformer.load_state_dict(pipe_transformer_state_dict, strict=False)

        # 加载 ref_image_proj_model 权重
        ref_proj_state_dict = {
            k.replace("ref_image_proj_model.", ""): v
            for k, v in state_dict.items() if k.startswith("ref_image_proj_model.")
        }
        self.ref_image_proj_model.load_state_dict(ref_proj_state_dict, strict=False)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 修正为 3 通道
        ])

    def generate(
        self,
        prompt="",
        src_img=None,
        ref_img=None,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=77,
        seed=None,
        makeup_level=1.0,
    ):
        if src_img is not None:
            src_img_tensor = self.transform(src_img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
            src_hidden_states = self.pipe.vae.encode(src_img_tensor).latent_dist.sample()
            src_hidden_states = (src_hidden_states - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            b, c, h, w = src_hidden_states.shape
            src_hidden_states = rearrange(src_hidden_states, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

            src_latent_image_ids = torch.zeros(h // 2, w // 2, 3)
            src_latent_image_ids[..., 0] = 1
            src_latent_image_ids[..., 1] = torch.arange(h // 2)[:, None]
            src_latent_image_ids[..., 2] = torch.arange(w // 2)[None, :]
            src_latent_image_ids = repeat(src_latent_image_ids, "h w c -> b (h w) c", b=b)
            src_latent_image_ids = src_latent_image_ids.to(self.device, dtype=torch.bfloat16)

        if ref_img is not None:
            ref_img_tensor = self.transform(ref_img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
            ref_hidden_states = self.pipe.vae.encode(ref_img_tensor).latent_dist.sample()
            ref_hidden_states = (ref_hidden_states - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            b, c, h, w = ref_hidden_states.shape
            ref_hidden_states = rearrange(ref_hidden_states, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            ref_hidden_states = self.ref_image_proj_model(ref_hidden_states)

            ref_latent_image_ids = torch.zeros(h // 2, w // 2, 3)
            ref_latent_image_ids[..., 0] = 2
            ref_latent_image_ids[..., 1] = torch.arange(h // 2)[:, None]
            ref_latent_image_ids[..., 2] = torch.arange(w // 2)[None, :]
            ref_latent_image_ids = repeat(ref_latent_image_ids, "h w c -> b (h w) c", b=b)
            ref_latent_image_ids = ref_latent_image_ids.to(self.device, dtype=torch.bfloat16)

            # shift
            ref_latent_image_ids[..., 2] -= (width // 16)

        generator = None if seed is None else torch.Generator(self.device).manual_seed(seed)

        images = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            src_hidden_states=src_hidden_states,
            ref_hidden_states=ref_hidden_states,
            cond_img_ids=[src_latent_image_ids, ref_latent_image_ids],
            max_sequence_length=max_sequence_length,
            makeup_level=makeup_level,
        ).images[0]
        return images

makeup_current_path = os.path.dirname(os.path.abspath(__file__))
weigths_current_path = os.path.join(folder_paths.models_dir, "stable_makeup")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

scheduler_list = ["DDIM",
    "Euler",
    "Euler a",
    "DDPM",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "Heun",
    "LCM",
    "LMS",
    "LMS Karras",
    "UniPC",
]

def get_sheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("Empty image input.")
        return ensure_pil_image(image[0])
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3) and tensor.shape[-1] != 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.contiguous()
        if tensor.dtype != torch.uint8:
            tensor = tensor.float().clamp(0, 1)
            array = (tensor.numpy() * 255).astype(np.uint8)
        else:
            array = tensor.numpy()
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        return Image.fromarray(array)
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError(f"Unsupported image type: {type(image)}")

def pil_to_tensor_batch(pil_img):
    array = np.array(pil_img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    tensor = torch.from_numpy(array)
    return tensor.unsqueeze(0)

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def resize_with_padding(pil_img, width, height):
    orig_w, orig_h = pil_img.size
    scale = min(width / orig_w, height / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height))
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, {"inner": (pad_x, pad_y, pad_x + new_w, pad_y + new_h)}

def get_face_bbox_and_crop(pil_img, net, expand=0.12, device="cuda"):
    """
    基于脸部语义分割得到人脸最小外接框，并从原图抠出该区域（正方形）。
    返回: (crop_pil, (x1, y1, x2, y2))；若未检测到脸，返回 (None, None)
    """
    W, H = pil_img.size
    with torch.no_grad():
        # 语义分割 -> 解析图
        logits = net(to_tensor()(pil_img)[None, ...].to(device))[0]
        parsing = logits.argmax(1).detach().cpu().numpy().astype(np.uint8)[0]

    # 聚合脸部区域
    mask = np.isin(parsing, face_ids).astype(np.uint8)
    if mask.sum() == 0:
        return None, None

    ys, xs = np.where(mask > 0)
    h_mask, w_mask = mask.shape
    scale_x = W / w_mask
    scale_y = H / h_mask

    x1m, x2m = xs.min(), xs.max()
    y1m, y2m = ys.min(), ys.max()

    x1 = int(np.floor(x1m * scale_x))
    x2 = int(np.ceil((x2m + 1) * scale_x)) - 1
    y1 = int(np.floor(y1m * scale_y))
    y2 = int(np.ceil((y2m + 1) * scale_y)) - 1

    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    dx = int(box_w * expand)
    dy = int(box_h * expand)
    x1e = max(0, x1 - dx)
    y1e = max(0, y1 - dy)
    x2e = min(W - 1, x2 + dx)
    y2e = min(H - 1, y2 + dy)

    # 强制正方形：以扩展框中心为中心，边长=max(宽,高)，并不越界
    cx = (x1e + x2e) / 2.0
    cy = (y1e + y2e) / 2.0
    side = max(x2e - x1e + 1, y2e - y1e + 1)
    side = min(side, W, H)  # 不能超过图像尺寸

    x1s = int(round(cx - side / 2))
    y1s = int(round(cy - side / 2))
    # 贴边裁剪，确保在图内且保持正方形
    x1s = max(0, min(x1s, W - side))
    y1s = max(0, min(y1s, H - side))
    x2s = x1s + side - 1
    y2s = y1s + side - 1

    crop = pil_img.crop((x1s, y1s, x2s + 1, y2s + 1))
    return crop, (x1s, y1s, x2s, y2s)

def to_tensor():
    return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
face_ids = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13]

class StableMakeup_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_path": ("STRING", {"default": "models/stable_makeup/FLUX.1-Kontext-dev"}),
                "makeup_model_path": ("STRING", {"default": "models/stable_makeup/Flux-Makeup-model/checkpoint.pt"}),
            }
        }

    RETURN_TYPES = ("MAKEUP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Stable_Makeup"

    def main_loader(self,flux_path,makeup_model_path):
        pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16, device="cuda")
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

        makeupAttnProcessor = {}
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks") or name.startswith("single_transformer_blocks"):
                makeupAttnProcessor[name] = MakeupFluxAttnProcessor().to(device="cuda", dtype=torch.bfloat16)
        transformer.set_attn_processor(makeupAttnProcessor)

        pipe.transformer = transformer
        pipe.to("cuda")
        wrappedPipe = WrappedPipe(pipe, makeup_model_path)
        
        # model
        n_classes = 19
        net = BiSeNet(n_classes=n_classes).cuda()
        net.load_state_dict(torch.load("models/stable_makeup/face-parsing.PyTorch/79999_iter.pth"))
        net.eval()

        # Unet= pipe.unet
        # vae=pipe.vae
        # text_encoder = pipe.text_encoder
        # id_encoder = ControlNetModel.from_unet(Unet)
        # pose_encoder = ControlNetModel.from_unet(Unet)
        # repo=os.path.join(makeup_current_path,"clip")
        # makeup_encoder = detail_encoder(Unet, clip,repo, "cuda", dtype=torch.float32)
        # makeup_state_dict = torch.load(makeup_encoder_path)
        # id_state_dict = torch.load(id_encoder_path)
        
        # id_encoder.load_state_dict(id_state_dict, strict=False)
        # pose_state_dict = torch.load(pose_encoder_path)
        # pose_encoder.load_state_dict(pose_state_dict, strict=False)
        # makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
        
        # del id_state_dict,makeup_state_dict,pose_state_dict
        # torch.cuda.empty_cache()
        
        # id_encoder.to("cuda")
        # pose_encoder.to("cuda")
        # makeup_encoder.to("cuda")
        # if lora!="none":
        #     lora_path = folder_paths.get_full_path("loras", lora)
        #     pipe.load_lora_weights(lora_path, adapter_name=lora_trigger_words)
        #     pipe.fuse_lora(adapter_names=[lora_trigger_words, ], lora_scale=lora_scale)
        # pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     sd15_config,
        #     safety_checker=None,
        #     unet=Unet,
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     controlnet=[id_encoder, pose_encoder],
        #     torch_dtype=torch.float32).to("cuda")
        # pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config)
        
        return ({"pipe":wrappedPipe,"net":net},)
        # return ({"pipe":pipe,"makeup_encoder":makeup_encoder},)

class StableMakeup_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_image": ("IMAGE",),
                "makeup_image": ("IMAGE",),
                "model": ("MAKEUP_MODEL",),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "makeup_level": ("FLOAT", {"default": 1.0 }),
                "post_level": ("FLOAT", {"default": 1.0 }),
                # "width": ("INT", {"default": 1024, "min": 256, "max": 1024, "step": 64, "display": "number"}),
                # "height": ("INT", {"default": 1024, "min": 256, "max": 1024, "step": 64, "display": "number"}),
                
               }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "face_crop", "makeup_crop")
    FUNCTION = "makeup_main"
    CATEGORY = "Stable_Makeup"
    
    def makeup_main(self, id_image, makeup_image, model,cfg, steps, makeup_level, post_level):
        width = height = 1024  # 强制为 1024x1024
        
        pipe=model.get("pipe")
        net=model.get("net")
        
        # pdb breakpoint for debugging: inspect id_image / makeup_image and other locals
        id_image_pil = ensure_pil_image(id_image) # id_image [1, 1024, 1024, 3] 'torch.Tensor' 0-1之间 torch.float32
        makeup_image_pil = ensure_pil_image(makeup_image)
        
        face_crop, face_bbox = get_face_bbox_and_crop(id_image_pil, net, expand=0.3, device="cuda")
        makeup_crop, makeup_bbox = get_face_bbox_and_crop(makeup_image_pil, net, expand=0.3, device="cuda")

        face_crop_ori = copy.deepcopy(face_crop)
        makeup_crop_ori = copy.deepcopy(makeup_crop)

        if face_crop is None or makeup_crop is None:
            fallback = id_image_pil.resize((width, height), Image.BILINEAR)
            return (pil2narry(fallback),)

        # 以 face_crop 的边长为基准，取最近的 64 倍数并限制在 [64, 1024]
        def _nearest_mul64(n):
            m = int(round(n / 64.0)) * 64
            return min(1024, max(64, m))
        face_side = _nearest_mul64(face_crop.size[0])
        makeup_side = _nearest_mul64(makeup_crop.size[0])
        # 忽略传入的参数，直接赋值
        width = height = face_side

        face_crop, face_pad_info = resize_with_padding(face_crop, face_side, face_side)
        makeup_crop, _ = resize_with_padding(makeup_crop, makeup_side, makeup_side)
        
        makeup_mask=net(to_tensor()(makeup_crop)[None,...].cuda())[0].argmax(1).cpu().numpy().astype(np.uint8)[0]
        mask_img = Image.fromarray(makeup_mask).resize((makeup_side, makeup_side), Image.NEAREST)
        mask = np.isin(np.array(mask_img), face_ids).astype(np.uint8)
        makeup_crop = Image.fromarray((np.array(makeup_crop) * mask[..., None]).astype(np.uint8))
        # pdb.set_trace()
        # pose_image = get_draw(id_image, size=width,dataname=dataname)
        # result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image,num_inference_steps=steps,pipe=pipe, guidance_scale=cfg)
        result_img = pipe.generate(
            prompt="makeup.",
            src_img=face_crop,
            ref_img=makeup_crop,
            height=height,
            width=width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            max_sequence_length=512,
            seed=42,
            makeup_level=makeup_level,
        )
        result_img = Image.fromarray((post_level * np.array(result_img) + (1 - post_level) * np.array(face_crop)).astype(np.uint8))

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            inner_box = face_pad_info["inner"]
            face_region = result_img.crop(inner_box)
            target_size = (x2 - x1 + 1, y2 - y1 + 1)  # (width, height)
            
            if face_region.size != target_size:
                face_region = face_region.resize(target_size, Image.BILINEAR)

            # --- 高级方案: 泊松融合 (Poisson Blending) 开始 ---
            try:
                # 1. 准备图像: PIL (RGB) -> OpenCV (BGR)
                # 源图像 (AI生成的脸部区域)
                src_img_cv = cv2.cvtColor(np.array(face_region), cv2.COLOR_RGB2BGR)
                # 目标图像 (原始背景图)
                dst_img_cv = cv2.cvtColor(np.array(id_image_pil), cv2.COLOR_RGB2BGR)

                # 2. 获取尺寸并创建蒙版
                tw, th = target_size  # (width, height)
                
                # 蒙版必须是单通道 (uint8), 告诉 seamlessClone 要复制 src_img_cv 中的哪些区域
                # 在这里, 我们要复制整个矩形区域
                mask = np.full((th, tw), 255, dtype=np.uint8)  # 注意: numpy 数组形状是 (height, width)

                # 3. 计算粘贴的中心点
                # 这是 src_img_cv 的中心点在 dst_img_cv (目标图) 上的坐标
                center = (x1 + tw // 2, y1 + th // 2)  # (center_x, center_y)

                # 4. 执行泊松融合 (关键步骤)
                # cv2.NORMAL_CLONE 会自动匹配 src 到 dst 的颜色和光照
                blended_img_cv = cv2.seamlessClone(src_img_cv, dst_img_cv, mask, center, cv2.NORMAL_CLONE)
                
                # 5. 转换回 PIL (BGR -> RGB)
                composed_img = Image.fromarray(cv2.cvtColor(blended_img_cv, cv2.COLOR_BGR2RGB))

            except cv2.error as e:
                # 如果 seamlessClone 出错 (比如边界问题), 回退到你原来的 Alpha Blending
                print(f"OpenCV seamlessClone 失败 (错误: {e}). 回退到 Alpha Blending...")
                
                base_face_region = id_image_pil.copy().crop((x1, y1, x2 + 1, y2 + 1))

                # 使用你建议的参数，但调大羽化范围
                tw, th = target_size
                feather = max(6, int(min(tw, th) * 0.1)) # 调大系数到 0.1
                yy, xx = np.ogrid[:th, :tw]
                dist = np.minimum(np.minimum(yy, th - 1 - yy), np.minimum(xx, tw - 1 - xx)).astype(np.float32)
                alpha = np.clip(dist / max(1, feather), 0.0, 1.0)
                edge_mask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
                edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))

                blended_face = Image.composite(face_region, base_face_region, edge_mask)
                composed_img = id_image_pil.copy()
                composed_img.paste(blended_face, (x1, y1))
            
            # --- 泊松融合 (Poisson Blending) 结束 ---

        else:
            composed_img = result_img
            
        image = pil2narry(composed_img)
        face_out = pil2narry(face_crop_ori)
        makeup_out = pil2narry(makeup_crop_ori)
        return (image, face_out, makeup_out)
        ##########################################

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            inner_box = face_pad_info["inner"]
            face_region = result_img.crop(inner_box)
            target_size = (x2 - x1 + 1, y2 - y1 + 1)
            if face_region.size != target_size:
                face_region = face_region.resize(target_size, Image.BILINEAR)

            composed_img = id_image_pil.copy()
            base_face_region = composed_img.crop((x1, y1, x2 + 1, y2 + 1))

            # 仅对粘贴区域的四边做羽化，避免硬边
            tw, th = target_size
            feather = max(6, int(min(tw, th) * 0.06))  # 可根据效果调 0.04~0.1
            yy, xx = np.ogrid[:th, :tw]
            # 修复 ValueError：使用可广播的两两最小值，避免 reduce 时产生不规则 object 数组
            dist = np.minimum(np.minimum(yy, th - 1 - yy), np.minimum(xx, tw - 1 - xx)).astype(np.float32)
            alpha = np.clip(dist / max(1, feather), 0.0, 1.0)
            edge_mask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
            edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))

            blended_face = Image.composite(face_region, base_face_region, edge_mask)
            composed_img.paste(blended_face, (x1, y1))
        else:
            composed_img = result_img
        image = pil2narry(composed_img)
        face_out = pil2narry(face_crop_ori)
        makeup_out = pil2narry(makeup_crop_ori)
        return (image, face_out, makeup_out)

NODE_CLASS_MAPPINGS = {
    "StableMakeup_LoadModel":StableMakeup_LoadModel,
    "StableMakeup_Sampler": StableMakeup_Sampler

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StableMakeup_LoadModel":"StableMakeup_LoadModel",
    "StableMakeup_Sampler": "StableMakeup_Sampler"
}
