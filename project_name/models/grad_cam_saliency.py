# gradcam_utils.py

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_gradcam(model, input_tensor, target_class=None, stream='rgb'):
    """
    Compute Grad-CAM for the TwoStreamDetector model.

    Args:
        model: Trained TwoStreamDetector model.
        input_tensor: Tuple (rgb_tensor, err_tensor) with shape (1, C, H, W).
        target_class: Class index to compute Grad-CAM for. If None, use predicted class.
        stream: One of 'rgb' or 'err' to specify which stream to visualize.

    Returns:
        heatmap: Numpy array of Grad-CAM heatmap (H, W).
    """

    model.eval()
    rgb_input, err_input = input_tensor
    rgb_input = rgb_input.requires_grad_(True)
    err_input = err_input.requires_grad_(True)

    def get_stream_and_input():
        if stream == 'rgb':
            feature_extractor = model.rgb_stream.blocks
            input_used = rgb_input
        elif stream == 'err':
            feature_extractor = model.err_stream.blocks
            input_used = err_input
        else:
            raise ValueError("Stream must be 'rgb' or 'err'")
        return feature_extractor, input_used

    features, input_used = get_stream_and_input()

    # Forward hook
    activations = []
    def forward_hook(module, inp, outp):
        activations.append(outp)

    handle = features[-1].register_forward_hook(forward_hook)

    # Forward pass
    with torch.enable_grad():
        _ = model(rgb_input, err_input)
        if target_class is None:
            pred = torch.argmax(model(rgb_input, err_input), dim=1)
        else:
            pred = torch.tensor([target_class]).to(input_used.device)

        score = model(rgb_input, err_input)[0, pred]
        score.backward()

    handle.remove()

    grads = input_used.grad
    target_activations = activations[0]  # Last conv block output
    pooled_grads = grads.mean(dim=[0, 2, 3])

    # Weighted sum of the activations
    for i in range(pooled_grads.shape[0]):
        target_activations[:, i, :, :] *= pooled_grads[i]

    heatmap = target_activations.mean(dim=1).squeeze()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap = heatmap / torch.max(heatmap)

    return heatmap.cpu().numpy()

def show_gradcam_on_image(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.40,
    show: bool = True,
):
    """
    Blend a Grad-CAM heat-map on top of the original RGB image and either
    display it (Matplotlib) or just return the result.

    Parameters
    ----------
    image_tensor : (3,H,W) torch.Tensor in [0,1]
    heatmap      : (H,W) numpy array – arbitrary range, normalised internally
    alpha        : opacity of the heat-map (0 = only image, 1 = only CAM)
    show         : if True (default) pop up with plt.imshow; otherwise just
                   return the uint8 overlay.
    """
    # ---- prepare background image ------------------------------------------------
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()  # → HWC, float
    if img.max() <= 1.0 + 1e-6:               # already [0,1]
        img = (img * 255.0).astype(np.uint8)
    else:                                     # assume [0,255]
        img = img.astype(np.uint8)

    # ---- prepare CAM -------------------------------------------------------------
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()           # (1,H,W) → (H,W)

    # resize to match the image (OpenCV wants WxH order!)
    cam = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # → [0,1]

    cam_uint8  = np.uint8(cam * 255)
    cam_color  = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR
    cam_color  = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)      # RGB uint8

    # ---- blend -------------------------------------------------------------------
    overlay = cv2.addWeighted(cam_color, alpha, img, 1.0 - alpha, 0)

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Grad-CAM Overlay")
        plt.show(block=False)

    return overlay  # uint8 RGB image

