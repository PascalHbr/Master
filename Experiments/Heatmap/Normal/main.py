from models import *
from tqdm import tqdm

from config import *
from dataset import get_dataset, DataLoader
from utils import *
import glob


def main(arg):
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Choose categories to plot
    categories_to_plot = None

    # Datasets
    model_names = ["slow", "x3d", "mvit", "mae"] if arg.model == "all" else [arg.model]
    for model_name in model_names:
        Dataset = get_dataset(arg.dataset)
        test_set = Dataset(mode='test', model=model_name, categories_to_plot=categories_to_plot)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Initialize model
        Model = get_model(model_name)
        model = Model(num_classes=test_set.num_classes, pretrained=True, freeze=False,
                      keep_head=True, device=device).to(device)
        model.eval()

        if arg.labels:
            # Iteration
            for step, (video, video_path, label, video_org) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    video = video.to(device)
                    video.requires_grad = True

                    # Get predictions
                    preds = model(video)
                    post_act = torch.nn.Softmax(dim=1)
                    preds = post_act(preds)
                    pred_classes = preds.topk(k=5).indices[0]
                    pred_class_names = [test_set.kinetics_labels[int(i)] for i in pred_classes]

                    if arg.labels:
                        label_category = "_".join(test_set.kinetics_labels[label.item()].split(" ")[:3]).replace("(", "").replace(")", "")
                        prediction_category = "_".join(pred_class_names[0].split(" ")[:3]).replace("(", "").replace(")", "")

                        for method in ["GuidedBackprop", "DeepLift", "Saliency"]:
                            label_heatmap = get_heatmap(model, video, label.item(), method)
                            prediction_heatmap = get_heatmap(model, video, pred_classes[0].item(), method)
                            save_heatmap(model_name, label_heatmap, video_path, method, "label", label_category)
                            save_heatmap(model_name, prediction_heatmap, video_path, method, "prediction", prediction_category)

        if arg.plot:
            for label in ["prediction", "label"]:
                for method in ["GuidedBackprop"]:
                    wandb.init(project='Heatmaps UCF', name=f"{model_name}_{label}_{method}", reinit=True)
                    heatmaps = sorted(glob.glob(f'/export/home/phuber/archive/Heatmap/ucf/{method}/{model_name}/*{label}*.npy'))
                    if categories_to_plot is not None:
                        heatmaps = [heatmap for heatmap in heatmaps if heatmap.split('/')[-1][2:-12].split("_")[0] in test_set.categories.keys()]
                    heatmap_categories = ["_".join(heatmap.split("_")[5:])[:-4] for heatmap in heatmaps]
                    plot_heatmaps(heatmaps, test_set, test_loader, categories_to_plot, heatmap_categories)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)