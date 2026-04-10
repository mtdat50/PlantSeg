import torch
import random
import glob
import cv2
import os
import shutil
import numpy as np

from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmengine.config import Config
from mmcv.transforms import Compose

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def visualize(model, pipeline, target_layers, image_path, device="cuda"):
    data = pipeline(
        dict(
            img_path=image_path,
            seg_map_path=None
        )
    )

    input_tensor = data["inputs"]
    input_tensor = input_tensor.unsqueeze(0).to(device).float()

    processed = input_tensor[0].cpu().numpy()
    processed = processed.transpose(1,2,0)
    processed = (processed - processed.min())/(processed.max()-processed.min())

    cam = EigenCAM(
        model=model,
        target_layers=target_layers,
    )

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0]

    cam_image = show_cam_on_image(
        processed,
        grayscale_cam,
        use_rgb=True
    )

    return cam_image


def grayscale_cam(model, image, target_layers, device="cuda"):
    image = image.unsqueeze(0).to(device).float()

    cam = EigenCAM(
        model=model,
        target_layers=target_layers,
    )

    cam = cam(input_tensor=image)
    cam = cam[0]

    return cam



if __name__ == "__main__":
    og_conf_file = "configs/segnext/segnext_mscan-t_10k_plantsegwheat.py"
    og_chkpt_file = "/home/mtdat/Downloads/ogwheat_iter_10000.pth"
    ham_conf_file = "configs/segnext/segnext_mscan-t_10k_wheat_ham.py"
    ham_chkpt_file = "/home/mtdat/Downloads/iter_10000.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    register_all_modules()

    og_cfg = Config.fromfile(og_conf_file)
    og_model = init_model(og_cfg, og_chkpt_file, device=device)
    og_model.eval()
    print(og_model)
    og_target_layers = [og_model.decode_head.align]

    ham_cfg = Config.fromfile(ham_conf_file)
    ham_model = init_model(ham_cfg, ham_chkpt_file, device=device)
    ham_model.eval()
    ham_target_layers = [ham_model.decode_head.align]

    pipeline_cfg = og_cfg.test_dataloader.dataset.pipeline
    # pipeline_cfg = [p for p in pipeline_cfg if p["type"] != "LoadAnnotations"]
    pipeline = Compose(pipeline_cfg)

    image_paths = glob.glob("data/plantsegwheat/images/test/*.jpg")
    # random.seed(42)
    # image_paths = random.choices(image_paths, k=50)

    # for image_path in image_paths:
    #     image_name = os.path.basename(image_path).split(".")[0]
    #     os.makedirs(f"visualize/{image_name}", exist_ok=True)
    #
    #     cam_image = visualize(og_model, pipeline, og_target_layers, image_path, device=device)
    #     cv2.imwrite(f"visualize/{image_name}/og.png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    #
    #     cam_image = visualize(ham_model, pipeline, ham_target_layers, image_path, device=device)
    #     cv2.imwrite(f"visualize/{image_name}/wheatham.png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    #
    #     mask_path = f"data/plantsegwheat/annotations/test/{image_name}.png"
    #     shutil.copy(mask_path, f"visualize/{image_name}/mask.png")
    #     shutil.copy(image_path, f"visualize/{image_name}/image.jpg")


    classes = (
        'wheat_bacterial_leaf_streak_(black_chaff)',
        'wheat_head_scab',
        'wheat_leaf_rust',
        'wheat_loose_smut',
        'wheat_powdery_mildew',
        'wheat_septoria_blotch',
        'wheat_stem_rust',
        'wheat_stripe_rust'
    )

    cnt = [0 for _ in range(len(classes))]
    og_total_acc = [0 for _ in range(len(classes))]
    ham_total_acc = [0 for _ in range(len(classes))]
    for image_path in image_paths:
        image_name = os.path.basename(image_path).split(".")[0]
        mask_path = f"data/plantsegwheat/annotations/test/{image_name}.png"
        data = pipeline(
            dict(
                img_path=image_path,
                seg_map_path=mask_path,
                reduce_zero_label=False,
                seg_fields=[],
                img_prefix=None,
                seg_prefix=None
            )
        )
        mask_tensor = data['data_samples'].gt_sem_seg.data.squeeze(0).numpy()

        cam = grayscale_cam(og_model, data['inputs'], og_target_layers, device='cuda')
        cam = cv2.resize(cam, (mask_tensor.shape[1], mask_tensor.shape[0]))
        hit = np.where(mask_tensor > 0, cam, np.zeros_like(cam))
        miss = cam - hit
        og_acc = hit.sum() / (cam.sum() + 1e-9)

        cam = grayscale_cam(ham_model, data['inputs'], ham_target_layers, device='cuda')
        cam = cv2.resize(cam, (mask_tensor.shape[1], mask_tensor.shape[0]))
        hit = np.where(mask_tensor > 0, cam, np.zeros_like(cam))
        miss = cam - hit
        ham_acc = hit.sum() / (cam.sum() + 1e-9)

        # print(f"{image_name}: og_acc={og_acc:.4f}, ham_acc={ham_acc:.4f}, %diff={(ham_acc - og_acc) / (og_acc + 1e-9) * 100:.2f}%")
        for i in range(len(classes)):
            if classes[i] in image_name:
                og_total_acc[i] += og_acc.item()
                ham_total_acc[i] += ham_acc.item()
                cnt[i] += 1
                break

    print("og   ham     %diff")
    for i in range(len(classes)):
        og_avg_acc = og_total_acc[i] / cnt[i] if cnt[i] > 0 else 0
        ham_avg_acc = ham_total_acc[i] / cnt[i] if cnt[i] > 0 else 0
        print(f"{og_avg_acc:.4f} {ham_avg_acc:.4f} {(ham_avg_acc - og_avg_acc) / (og_avg_acc + 1e-9) * 100:.2f}%")



'''
 (block4): ModuleList(
    (0-1): 2 x MSCABlockWithHam(
      (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (attn): AttentionModule(
        (proj_1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (activation): GELU(approximate='none')
        (spatial_gating_unit): MSCAAttention(
          (conv0): Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256)
          (conv0_1): Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=256)
          (conv0_2): Conv2d(256, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=256)
          (conv1_1): Conv2d(256, 256, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5), groups=256)
          (conv1_2): Conv2d(256, 256, kernel_size=(11, 1), stride=(1, 1), padding=(5, 0), groups=256)
          (conv2_1): Conv2d(256, 256, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=256)
          (conv2_2): Conv2d(256, 256, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=256)
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (proj_2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (drop_path): DropPath()
      (norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (mlp): Mlp(
        (fc1): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
        (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
        (act): GELU(approximate='none')
        (fc2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (patch_embed5): OverlapPatchEmbed(
    (proj): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (block5): ModuleList(
    (0): MSCABlockWithHam(
      (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (attn): AttentionModule(
        (proj_1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (activation): GELU(approximate='none')
        (spatial_gating_unit): Hamburger(
          (ham_in): ConvModule(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (ham): NMF2D()
          (ham_out): ConvModule(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (gn): GroupNorm(32, 256, eps=1e-05, affine=True)
          )
        )
        (proj_2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (drop_path): DropPath()
      (norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (mlp): Mlp(
        (fc1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        (act): GELU(approximate='none')
        (fc2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (norm5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
)
'''
