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

def show_gradcam_on_image(image_tensor, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image tensor.

    Args:
        image_tensor: (3, H, W) torch.Tensor in range [0, 1]
        heatmap: Grad-CAM heatmap (H, W) in numpy
    """
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    image = np.uint8(255 * image)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + image

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img[..., ::-1])  # BGR to RGB
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()
