import torch
import numpy as np
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random
import os


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


def make_visualizations(frames, predictions, labels, directory, epoch, inference, target):
    size = frames.shape[1]
    frames = (255 * frames.cpu().detach().numpy()).astype(np.uint8)
    bbox_predictions = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(size * predictions.cpu().detach().numpy())]
    bbox_labels = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(size * labels.cpu().detach().numpy())]
    for i, frame in enumerate(frames):
        cv2.rectangle(frame, bbox_predictions[i][0], bbox_predictions[i][1], color=(255, 0, 0), thickness=1)
        cv2.rectangle(frame, bbox_labels[i][0], bbox_labels[i][1], color=(0, 255, 0), thickness=1)

    # Create PDF
    if inference:
        save_name = directory + 'summary.pdf'
    else:
        save_name = directory + str(epoch) + '_results.pdf'

    with PdfPages(save_name) as pdf:
        if inference or target == 'all':
            fig_head, axs_head = plt.subplots(8, 8, figsize=(15, 15))
            fig_head.suptitle("Overview", fontsize="x-large")
            for i in range(8):
                for j in range(8):
                    idx = i * 8 + j
                    axs_head[i, j].imshow(frames[idx])
                    axs_head[i, j].axis('off')
        else:
            fig_head, axs_head = plt.subplots(4, 4, figsize=(15, 15))
            fig_head.suptitle("Overview", fontsize="x-large")
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    axs_head[i, j].imshow(frames[idx])
                    axs_head[i, j].axis('off')
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


def save_embedding(embedding, video_path, dataset, name):
    if not os.path.exists(f'../../../../archive/embeddings/{dataset}/{name}/{video_path[0].split("/")[0]}'):
        os.makedirs(f'../../../../archive/embeddings/{dataset}/{name}/{video_path[0].split("/")[0]}')
    np.save(f'../../../../archive/embeddings/{dataset}/{name}/{video_path[0]}.npy', embedding[0].detach().cpu().numpy())

