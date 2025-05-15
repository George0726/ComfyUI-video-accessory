import os
import torch
import folder_paths
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from nodes import MAX_RESOLUTION
import comfy
import torch.nn.functional as F
import json
import av
from fractions import Fraction
import io
import cv2
import model_management
from comfy.utils import ProgressBar, common_upscale
import numpy as np
from einops import rearrange
import ast
import copy

import nodes
import torch
import numpy as np
from einops import rearrange
import comfy.model_management


# PIL to Tensor
def pil2tensor(image):
  return torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def decode_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        raise NotImplementedError

    elif encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
        try:
            image = Image.open(BytesIO(base64.b64decode(encoding)))
            return image
        except Exception as e:
            raise Exception from e
        
    elif len(encoding) > 0 and os.path.exists(encoding):
        try:
            image = Image.open(encoding)
            return image
        except Exception as e:
            raise Exception from e
    
    else:
        raise Exception(f'VideoAcc_LoadImage, not a  valid input format ==> {type(encoding), encoding}')


def decode_to_video(encoding):

    frames = []  # List to store frames
    if encoding.startswith("http://") or encoding.startswith("https://"):
        raise NotImplementedError

    elif encoding.startswith("data:video/"):
        encoding = encoding.split(";")[1].split(",")[1]
        try:
            np_video = np.frombuffer(encoding, dtype=np.uint8)

            # Decode video into OpenCV video stream
            video = cv2.VideoCapture()
            video.open(cv2.IMMEMORY, np_video)
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))  # Append frame to list
                # frame_count += 1

            video.release()
            return frames
        except Exception as e:
            raise Exception from e
        
    elif len(encoding) > 0 and os.path.exists(encoding):
        try:
            video = cv2.VideoCapture(encoding)

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))  # Append frame to list
                # frame_count += 1

            video.release()
            return frames
        except Exception as e:
            raise Exception from e
    
    else:
        raise Exception(f'VideoAcc_LoadImage, not a  valid input format ==> {type(encoding), encoding}')

MAX_RESOLUTION = nodes.MAX_RESOLUTION


CAMERA_DICT = {
    "base_T_norm": 0.4,              # Translation scale factor
    "base_angle": np.pi / 6,         # Rotation scale factor (60 degrees)

    # ─── Static ───
    "Static": {
        "angle": [0., 0., 0.],       # No rotation
        "T":     [0., 0., 0.]        # No translation
    },

    # ─── Rotations (Euler angles: Pitch=X, Yaw=Y, Roll=Z) ───
    "Tilt Up":       { "angle": [1., 0., 0.],  "T": [0., 0., 0.] },  # Pitch up
    "Tilt Down":     { "angle": [-1., 0., 0.], "T": [0., 0., 0.] },  # Pitch down
    "Pan Left":      { "angle": [0., 1., 0.],  "T": [0., 0., 0.] },  # Yaw left
    "Pan Right":     { "angle": [0., -1., 0.], "T": [0., 0., 0.] },  # Yaw right
    "Roll Left":     { "angle": [0., 0., 1.],  "T": [0., 0., 0.] },  # Roll counter-clockwise
    "Roll Right":    { "angle": [0., 0., -1.], "T": [0., 0., 0.] },  # Roll clockwise

    # ─── Translations (world space movement) ───
    "Truck Left":    { "angle": [0., 0., 0.],  "T": [-1., 0., 0.] },  # Move left
    "Truck Right":   { "angle": [0., 0., 0.],  "T": [1., 0., 0.] },   # Move right
    "Crane Up":      { "angle": [0., 0., 0.],  "T": [0., 1., 0.] },   # Move up
    "Crane Down":    { "angle": [0., 0., 0.],  "T": [0., -1., 0.] },  # Move down
    "Dolly In":      { "angle": [0., 0., 0.],  "T": [0., 0., 2.] },   # Move forward
    "Dolly Out":     { "angle": [0., 0., 0.],  "T": [0., 0., -2.] },  # Move backward

    # ─── Zoom (handled via fx/fy scaling in code) ───
    "Zoom In":       { "angle": [0., 0., 0.],  "T": [0., 0., 0.] },
    "Zoom Out":      { "angle": [0., 0., 0.],  "T": [0., 0., 0.] },
}

def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu'):

    def get_relative_pose(cam_params):
        """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
        """
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        c2w_mat = np.array(entry[7:]).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        indexing='ij'
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def get_camera_motion(angle, T, speed, n=81, speed_type = "steady", initial_angle=(0., 0., 0.), initial_location=(0., 0., 0.)):
    def compute_R_form_rad_angle(angles):
        theta_x, theta_y, theta_z = angles
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
    RT = []


    initial_R = compute_R_form_rad_angle(initial_angle)
    initial_T = np.array(initial_location).reshape(3, 1)

    for i in range(n):
        if speed_type == "steady":
            speed_factor = 1.0
        elif speed_type == "accelerate":
            speed_factor = 1.0 + (i / n) * 1.0
        else:
            speed_factor = 2.0 - (i / n) * 1.0

        _angle = (i/n)*speed_factor*speed*(CAMERA_DICT["base_angle"])*angle
        delta_R = compute_R_form_rad_angle(_angle)
        _T=(i/n)*speed_factor*speed*(CAMERA_DICT["base_T_norm"])*(T.reshape(3,1))

        # Compose final RT with initial
        R = delta_R @ initial_R
        T_total = initial_T + _T

        _RT = np.concatenate([R, T_total], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT



def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def combine_camera_motion(RT_0, RT_1):
    RT = copy.deepcopy(RT_0[-1])
    R = RT[:,:3]
    R_inv = RT[:,:3].T
    T =  RT[:,-1]

    temp = []
    for _RT in RT_1:
        _RT[:,:3] = np.dot(_RT[:,:3], R)
        _RT[:,-1] =  _RT[:,-1] + np.dot(np.dot(_RT[:,:3], R_inv), T) 
        temp.append(_RT)

    RT_1 = np.stack(temp)

    return np.concatenate([RT_0, RT_1], axis=0)

class VideoAcc_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING", {"multiline": True})},
                "optional": {"invert_mask": ("BOOLEAN", {"default": True}),
                             }}
    
    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "VideoAcc Nodes"
    FUNCTION = "load_image"

    def load_image(self, image, invert_mask = True):
        img = decode_to_image(image)
        w,h = img.size

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            if invert_mask:
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.from_numpy(mask)
        else:
            mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)

class VideoAcc_LoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING", {"multiline": True})},
                "optional": {"invert_mask": ("BOOLEAN", {"default": True}),
                             }}
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "VideoAcc Nodes"
    FUNCTION = "load_video"

    def load_video(self, image, invert_mask = True):
        img_list = decode_to_video(image)
        video_tensors = [ pil2tensor(_) for _ in img_list]
        return (torch.concat(video_tensors,dim=0),)

class VideoAcc_imageSize:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
        "max_length": ("INT",),
        "step": ("INT",),
      }
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width_int", "height_int")
  OUTPUT_NODE = True
  FUNCTION = "image_width_height"

  CATEGORY = "EasyUse/Image"

  def image_width_height(self, image, max_length=1024,step=16):
    _, raw_H, raw_W, _ = image.shape
    ratio = max_length/max(raw_W,raw_H)

    width = int(raw_W*ratio//step*step)
    height = int(raw_H*ratio//step*step)

    if width is not None and height is not None:
      result = (width, height)
    else:
      result = (0, 0)
    return {"ui": {"text": "Width: "+str(width)+" , Height: "+str(height)}, "result": result}
  
class VideoAcc_ImageResizeAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "Ultralytics/Utils"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

        return(outputs, outputs.shape[2], outputs.shape[1],)

class VideoAcc_SaveMP4:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.prefix_append = ""

    methods = {"default": "libx264" ,"h265": "libx265", "vp9": "libvpx-vp9"}

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "CRF": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Lower CRF means higher quality"}),
                     "method": (list(s.methods.keys()),),
                     },
                "optional": {"base64_only": ("BOOLEAN", {"default": False}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    FUNCTION = "save_video"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("save_path", "base64")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False)

    CATEGORY = "image/animation"

    def save_video(self, images, fps, filename_prefix, CRF, method, base64_only=False, prompt=None, extra_pnginfo=None):
        codec = self.methods.get(method, "libx264")  # Default to H.264

        if os.path.isdir("/".join(filename_prefix.split("/")[:-1])):
            filename = filename_prefix
            full_output_folder, subfolder, counter = '/', '', 0
            file = f"{filename}.mp4" if not filename.endswith(".mp4") else filename
        else:
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

            file = f"{filename}_{counter:05}.mp4"

        width, height = images[0].shape[1], images[0].shape[0]

        # In-memory buffer
        video_buffer = io.BytesIO()

        with av.open(video_buffer, mode="w", format="mp4") as container:
            stream = container.add_stream(codec, rate=Fraction(round(fps * 1000), 1000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {'crf': str(CRF), 'preset': 'ultrafast'}

            if prompt is not None:
                container.metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    container.metadata[x] = json.dumps(extra_pnginfo[x])

            for image in images:
                array = np.clip((image.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            container.mux(stream.encode())

        video_buffer.seek(0)
        if not base64_only:
            output_file_path = os.path.join(full_output_folder, file)
            with open(output_file_path, "wb") as f:
                f.write(video_buffer.getvalue())
            base64_video = ""
        else:
            base64_video = base64.b64encode(video_buffer.read()).decode("utf-8")

        return {"ui": {'text': (os.path.join(full_output_folder, file), base64_video)}, "result": (os.path.join(full_output_folder, file), base64_video)}

class VideoAcc_ImageUpscaleVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "images": ("IMAGE",),
                              "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Ultralytics/image"
    DESCRIPTION = """
        Same as ComfyUI native model upscaling node,  
        but allows setting sub-batches for reduced VRAM usage.
    """
    def upscale(self, upscale_model, images, per_batch):
        
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = images.movedim(-1,-3)
        
        steps = in_img.shape[0]
        pbar = ProgressBar(steps)
        t = []
        
        for start_idx in range(0, in_img.shape[0], per_batch):
            sub_images = upscale_model(in_img[start_idx:start_idx+per_batch].to(device))
            t.append(sub_images.cpu())
            # Calculate the number of images processed in this batch
            batch_count = sub_images.shape[0]
            # Update the progress bar by the number of images processed in this batch
            pbar.update(batch_count)
        upscale_model.cpu()
        
        t = torch.cat(t, dim=0).permute(0, 2, 3, 1).cpu()

        return (t,)

class VideoAcc_CameraTrajectoryRecam:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cam_type":("INT",{"default":1, "min": 0, "max": 10, "step":1}),
                "fx":("FLOAT",{"default":0.474812461, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":0.844111024, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 81, "step": 4}),
            },
        }

    RETURN_TYPES = ("LATENT","INT","INT","INT")
    RETURN_NAMES = ("WAN_CAMERA_EMBEDDING","width","height","length")
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self, cam_type, fx, fy, cx, cy, width, height,length):
        """
        Use Camera trajectory as extrinsic parameters from ReCamMaster to calculate Plücker embeddings (Sitzmannet al., 2021)
        index from 1 to 10 represent: Pan Right, Pan Left, Tilt Up, Tilt Down, Zoom In, Zoom Out;
        Translate Up (with rotation), Translate Down (with rotation), Arc Left (with rotation), Arc Right (with rotation)
        https://github.com/KwaiVGI/ReCamMaster
        """
        file_path = __file__
        folder_path = os.path.dirname(file_path)
        tgt_camera_path = os.path.join(folder_path,"camera_extrinsics.json")
        # camera_extrinsics is from https://github.com/KwaiVGI/ReCamMaster/blob/main/example_test_data/cameras/camera_extrinsics.json
        with open(tgt_camera_path, 'r') as file:
            cam_data = json.load(file)

        interval = 80 // (length -1) 
        cam_idx = list(range(81))[::interval]
        assert len(cam_idx) == length, "Wrong video length"

        traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cam_type):02d}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
        camera_pose_list = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            camera_pose_list.append(c2w)
        trajs=[]

        for cp in camera_pose_list:
            traj=[fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            traj.extend(cp[3])
            trajs.append(traj)

        cam_params = np.array([[float(x) for x in pose] for pose in trajs])
        cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
        control_camera_video = process_pose_params(cam_params, width=width, height=height)
        control_camera_video = control_camera_video.permute([3, 0, 1, 2]).unsqueeze(0).to(device=comfy.model_management.intermediate_device())

        control_camera_video = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)

        # Reshape, transpose, and view into desired shape
        b, f, c, h, w = control_camera_video.shape
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)

        return (control_camera_video, width, height, length)

class VideoAcc_CameraTrajectoryAdvance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":([_ for _ in CAMERA_DICT] + ["Customize"],{"default":"Static"}),
                "fx":("FLOAT",{"default":1.0, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":1.0, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 81, "step": 4}),
            },
            "optional":{
                "speed":("FLOAT",{"default":1.0, "min": 0, "max": 10.0, "step": 0.1}),
                "speed_type":(["steady", "accelerate", "decelerate"],{"default":"steady"}),
                "initial_angle": ("STRING", {"default": "[0.0, 0.0, 0.0]"}),  # Euler angles in radians
                "initial_location": ("STRING", {"default": "[0.0, 0.0, 0.0]"}),  # XYZ coordinates
                "zoom_factor":("FLOAT",{"default":0, "min": 0, "max": 1, "step": 0.01}),
                "camera_pose_1":([_ for _ in CAMERA_DICT] + [""],{"default":""}),
                "camera_pose_2":([_ for _ in CAMERA_DICT] + [""],{"default":""}),
                "camera_pose_3":([_ for _ in CAMERA_DICT] + [""],{"default":""}),
                "camera_pose_4":([_ for _ in CAMERA_DICT] + [""],{"default":""}),
                "customize_R": ("STRING", {"default": "[0.0, 0.0, 0.0]"}),
                "customize_T": ("STRING", {"default": "[0.0, 0.0, 0.0]"}),
            }

        }

    RETURN_TYPES = ("WAN_CAMERA_EMBEDDING","INT","INT","INT")
    RETURN_NAMES = ("camera_embedding","width","height","length")
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self, camera_pose, fx, fy, cx, cy, width, height, length, speed=1.0, speed_type="steady", initial_angle="[0.0, 0.0, 0.0]", initial_location="[0.0, 0.0, 0.0]",  zoom_factor=1.0, camera_pose_1="",camera_pose_2="",camera_pose_3="",camera_pose_4="",customize_R="[0.0, 0.0, 0.0]", customize_T="[0.0, 0.0, 0.0]",):
        """
        Use Camera trajectory as extrinsic parameters from ReCamMaster to calculate Plücker embeddings (Sitzmannet al., 2021)
        index from 1 to 10 represent: Pan Right, Pan Left, Tilt Up, Tilt Down, Zoom In, Zoom Out;
        Translate Up (with rotation), Translate Down (with rotation), Arc Left (with rotation), Arc Right (with rotation)
        https://github.com/KwaiVGI/ReCamMaster
        """
        camera_pose_list =[camera_pose]
        for pose in [camera_pose_1, camera_pose_2, camera_pose_3, camera_pose_4]:
            if len(pose) == 0:
                break
            camera_pose_list.append(pose)

        initial_angle = np.array(ast.literal_eval(initial_angle), dtype=np.float32)
        initial_location = np.array(ast.literal_eval(initial_location), dtype=np.float32)
        customize_R = np.array(ast.literal_eval(customize_R), dtype=np.float32)
        customize_T = np.array(ast.literal_eval(customize_T), dtype=np.float32)

        length_remain = 0
        pose_compose = []
        for idx, pose in enumerate(camera_pose_list):
            if idx < len(camera_pose_list) -1:
                target_length = length // len(camera_pose_list)
            else:
                target_length = length - length_remain
            pose_compose += [pose for _ in range(target_length)]
            length_remain += target_length


            angle = np.array(CAMERA_DICT[pose]["angle"] if pose != "Customize" else customize_R)
            T = np.array(CAMERA_DICT[pose]["T"] if pose != "Customize" else customize_T)
            RT = get_camera_motion(angle, T, speed, target_length, speed_type=speed_type,
                initial_angle=initial_angle,
                initial_location=initial_location
            )
            if idx > 0:
                RT_all = combine_camera_motion(RT_all, RT)
            else:
                RT_all = RT
                    
        trajs=[]
        scale = 1.0
        for i, cp in enumerate(RT_all.tolist()):
            if pose_compose[i] == "Zoom In":
                scale += (1 / length) * zoom_factor   # progressively zoom in to 130%
            elif pose_compose[i] == "Zoom Out":
                scale -= (1 / length) * zoom_factor   # progressively zoom out to 70%

            traj=[scale * fx,scale * fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            traj.extend([0,0,0,1])
            trajs.append(traj)

        cam_params = np.array([[float(x) for x in pose] for pose in trajs])
        cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
        control_camera_video = process_pose_params(cam_params, width=width, height=height)
        control_camera_video = control_camera_video.permute([3, 0, 1, 2]).unsqueeze(0).to(device=comfy.model_management.intermediate_device())

        control_camera_video = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)

        # Reshape, transpose, and view into desired shape
        b, f, c, h, w = control_camera_video.shape
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)

        return (control_camera_video, width, height, length)

        
NODE_CLASS_MAPPINGS = {
    "VideoAcc_LoadImage": VideoAcc_LoadImage,
    "VideoAcc_imageSize": VideoAcc_imageSize,
    "VideoAcc_ImageResizeAdvanced": VideoAcc_ImageResizeAdvanced,
    "VideoAcc_SaveMP4":VideoAcc_SaveMP4,
    "VideoAcc_LoadVideo": VideoAcc_LoadVideo,
    "VideoAcc_ImageUpscaleVideo":VideoAcc_ImageUpscaleVideo,
    "VideoAcc_CameraTrajectoryRecam": VideoAcc_CameraTrajectoryRecam,
    "VideoAcc_CameraTrajectoryAdvance": VideoAcc_CameraTrajectoryAdvance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAcc_LoadImage": "VideoAcc Load Image(URL, Base64, Path)",
    "VideoAcc_imageSize": "VideoAcc imageSize",
    "VideoAcc_ImageResizeAdvanced": "VideoAcc Image Resize Advanced",
    "VideoAcc_SaveMP4": 'VideoAcc Save video files',
    "VideoAcc_LoadVideo": 'VideoAcc Load Video(URL, Base64, Path)',
    "VideoAcc_ImageUpscaleVideo": 'VideoAcc upscale video',
    "VideoAcc_CameraTrajectoryRecam": "VideoAcc load default camera trajectory from ReCam",
    "VideoAcc_CameraTrajectoryAdvance": "VideoAcc Advanced Camera trajectory",
}

