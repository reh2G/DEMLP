import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = {0: 'Healthy', 1: 'Stone'}

# ─── Compute Grad-CAM heatmap from the last convolutional layer
#
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # If all values are zero, deal with division by zero
    if tf.math.is_nan(heatmap[0, 0]):
        heatmap = tf.zeros_like(heatmap)
    
    return heatmap.numpy()

# ─── Overlay Grad-CAM heatmap on image and save with prediction info
#
def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", confidence=None, predicted_label=None, true_label=None, alpha=0.4):
    # Prepare the original image for display
    if np.max(img_array) <= 1.0:
        img_display = np.uint8(255 * img_array)
    else:
        img_display = np.uint8(img_array)

    if len(img_display.shape) == 3 and img_display.shape[-1] == 1:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    elif len(img_display.shape) == 2:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

    # Build viridis heatmap overlay
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))

    viridis = plt.colormaps.get_cmap("viridis")
    heatmap_colored = viridis(heatmap_resized)[:, :, :3]  # RGB, range [0, 1]
    heatmap_colored = np.uint8(255 * heatmap_colored)

    superimposed_img = (heatmap_colored * alpha + img_display * (1 - alpha)).astype(np.uint8)

    # Save with matplotlib (image + colorbar)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(superimposed_img)

    # Add the viridis colorbar using the heatmap intensity values
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Intensidade', fontsize=10)

    ax.axis('off')

    if confidence is not None and predicted_label is not None and true_label is not None:
        pred_name = CLASS_NAMES.get(predicted_label, str(predicted_label))
        true_name = CLASS_NAMES.get(true_label, str(true_label))
        
        title = f"Predição: {pred_name} ({confidence:.2%})"
        subtitle = f"Diagnóstico real: {true_name}"
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes, 
                fontsize=11, ha='center', va='top',
                color='green' if predicted_label == true_label else 'red')

    plt.tight_layout()
    plt.savefig(cam_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
