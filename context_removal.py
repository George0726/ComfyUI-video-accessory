
import torch
import numpy as np
import torch.nn.functional as F
from scipy import ndimage

def draw_red_contour_on_tensor(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    """
    Draw red contour on the image tensor using the edge of the binary mask.

    Args:
        image_tensor (torch.Tensor): (H, W, 3), values in [0, 1] or [0, 255]
        mask_tensor (torch.Tensor): (H, W), binary or grayscale

    Returns:
        torch.Tensor: image tensor with red contour added
    """
    # Ensure image is in uint8 [0, 255]
    if image_tensor.max() <= 1.0:
        image_np = (image_tensor.numpy() * 255).astype(np.uint8)
    else:
        image_np = image_tensor.numpy().astype(np.uint8)

    # Convert mask to binary
    mask_np = (mask_tensor.numpy() > 0.5).astype(np.uint8)

    # Extract edges using binary erosion
    eroded = ndimage.binary_erosion(mask_np)
    edges = mask_np - eroded

    if thickness > 1:
        structure = ndimage.generate_binary_structure(2, 1)
        edges = ndimage.binary_dilation(edges, structure=structure, iterations=thickness - 1)

    # Get edge coordinates
    edge_coords = np.argwhere(edges)
    
    # Draw red on edge pixels
    for y, x in edge_coords:
        image_np[y, x] = [255, 0, 0]  # Red

    # Return as torch tensor
    return torch.from_numpy(image_np).float() / 255.0  # Normalize back to [0, 1] if needed


class Kontext_removal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "image": ("IMAGE",),
                            "mask": ("MASK",),
                            "thickness":("INT", {"default": 1}),
                             },
                "optional": {
                    "method":(['QWEN','Kontext'], {"default": 'Kontext'}),
                }
                                
                             }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Seekoo Nodes"
    FUNCTION = "removal_control"

    def removal_control(self, image, mask, thickness = 1, method = 'Kontext'):
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        output_images = []
        for img,msk in zip(image, mask):
            if method == "Kontext":
                results  = draw_red_contour_on_tensor(img,msk,thickness)

            output_images.append(results)
        
        image = torch.stack(output_images)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "VideoAcc_Kontext_removal": Kontext_removal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAcc_Kontext_removal": "VideoAcc Removal Context",
}

