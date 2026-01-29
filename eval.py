import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.attn_processor import MakeupFluxAttnProcessor, ImageProjModel
from torchvision import transforms
from einops import rearrange, repeat
import cv2
import numpy as np
import os
import albumentations as A
import argparse
from tqdm import tqdm
import re
from mask_model import BiSeNet
from pathlib import Path


N_CLASSES = 19
FACE_IDS = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13]
WEIGHTS_PATH = Path(r"model/79999_iter.pth")


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
            transforms.Normalize([0.5], [0.5]),
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
        ).images[0]
        return images


def build_transform():
    """把PIL图像变成网络输入Tensor：Resize到(512,512)+Normalize。"""
    return transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    
@torch.no_grad()
def predict_face_mask(net, pil_img, device):
    """
    用BiSeNet预测语义分割并生成二值脸部mask。
    返回:
      mask_u8: (H,W) uint8，取值 {0,255}
      masked_pil: PIL.Image，原图乘mask后的抠图（非必须但你原代码有这步）
    """
    w, h = pil_img.size

    x = build_transform()(pil_img).unsqueeze(0).to(device)  # (1,3,512,512)

    # 兼容你原来的写法：net(...)[0].argmax(1)
    out = net(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    pred = out.argmax(1).detach().cpu().numpy().astype(np.uint8)[0]  # (512,512)

    # resize回原图大小
    pred_img = Image.fromarray(pred).resize((w, h), Image.NEAREST)
    pred_np = np.array(pred_img)

    mask01 = np.isin(pred_np, FACE_IDS).astype(np.uint8)  # (H,W) in {0,1}
    # mask_u8 = (mask01 * 255).astype(np.uint8)
    mask_u8 = pred_np.astype(np.uint8)

    # 按你原逻辑：makeup_crop * mask[...,None]
    img_np = np.array(pil_img).astype(np.uint8)
    masked_np = (img_np * mask01[..., None]).astype(np.uint8)
    masked_pil = Image.fromarray(masked_np)

    return mask_u8, masked_pil

def main():
    def create_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model',type=str)

        return parser

    args = create_argparser().parse_args()
    
    src_img_dir = 'eval_img/src'
    ref_img_dir = 'eval_img/ref'

    ckpt_path = args.model

    device = "cuda"
    base_path = "model/FLUX.1-Kontext-dev"

    # 初始化 pipeline
    pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device=device
    )

    makeupAttnProcessor = {}
    for name in transformer.attn_processors.keys():
        if name.startswith("transformer_blocks") or name.startswith("single_transformer_blocks"):
            makeupAttnProcessor[name] = MakeupFluxAttnProcessor().to(device=device, dtype=torch.bfloat16)
    transformer.set_attn_processor(makeupAttnProcessor)

    pipe.transformer = transformer
    pipe.to(device)
    
    # 初始化mask model
    net = BiSeNet(n_classes=N_CLASSES).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # 初始化 WrappedPipe
    wrappedPipe = WrappedPipe(pipe, ckpt_path)

    height, width = 1024, 1024
    face_ids = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13]

    output_dir = f'output_eval'

    os.makedirs(output_dir,exist_ok=True)

    src_img_list = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for i, src_name in enumerate(tqdm(src_img_list, desc="Processing src images")):
        src_path = os.path.join(src_img_dir, src_name)
        ref_path = os.path.join(ref_img_dir, src_name)

        src_img = Image.open(src_path).resize((height, width)).convert("RGB")
        ref_img = Image.open(ref_path).resize((height, width)).convert("RGB")
        mask_u8, masked_pil = predict_face_mask(net, ref_img, device)

        ref_img = np.array(ref_img)
        ref_mask = mask_u8
        face_mask = np.isin(ref_mask, face_ids).astype(np.uint8)
        ref_img = ref_img * face_mask[..., None]
        ref_img = Image.fromarray(ref_img)

        result = wrappedPipe.generate(
            prompt="makeup.",
            src_img=src_img,
            ref_img=ref_img,
            height=height,
            width=width,
            guidance_scale=2.5,
            num_inference_steps=25,
            max_sequence_length=512,
            seed=42,
        )

        out_name = os.path.splitext(src_name)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        result.save(out_path)


if __name__ == "__main__":
    main()
