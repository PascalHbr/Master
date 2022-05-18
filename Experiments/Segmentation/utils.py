import torch
import numpy as np
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir, device):
    model.load_state_dict(torch.load(model_save_dir + '/parameters', map_location=device))
    return model


def make_visualizations(frames, predictions, labels, directory, epoch, inference, save=False):
    frames = (255 * frames.cpu().detach().numpy()).astype(np.uint8)
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # Create PDF
    if inference:
        save_name = directory + '_summary.pdf'
    else:
        save_name = directory + str(epoch) + '_results.pdf'

    with PdfPages(save_name) as pdf:
        fig_head, axs_head = plt.subplots(3, 9, figsize=(15, 15))
        fig_head.suptitle("Overview", fontsize="x-large")
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                axs_head[i, 3*j].imshow(frames[idx], cmap='gray')
                axs_head[i, 3*j+1].imshow(labels[idx], cmap='gray')
                axs_head[i, 3*j+2].imshow(predictions[idx], cmap='gray')
                axs_head[i, 3*j].axis('off')
                axs_head[i, 3*j+1].axis('off')
                axs_head[i, 3*j+2].axis('off')

        if save:
            pdf.savefig(fig_head)

        fig_head.canvas.draw()
        w, h = fig_head.canvas.get_width_height()
        results = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        plt.close('all')

    return results


def set_random_seed(id=42):
    # Set random seeds
    random.seed(id)
    torch.manual_seed(id)
    torch.cuda.manual_seed(id)
    np.random.seed(id)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(id)
    rng = np.random.RandomState(id)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)