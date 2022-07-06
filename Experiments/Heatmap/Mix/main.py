from models import *
from tqdm import tqdm

from config import *
from dataset import get_dataset, DataLoader
from utils import *
import glob


def main(arg):
    # Set random seed
    set_random_seed(id=42)

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Choose categories to plot
    categories_to_plot = None

    # Datasets
    model_names = ["slow", "x3d", "mvit", "mae"] if arg.model == "all" else [arg.model]
    for model_name in model_names:
        Dataset = get_dataset(arg.dataset)
        test_set = Dataset(mode='test', model=model_name, mix=arg.mix, categories_to_plot=categories_to_plot)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Initialize model
        Model = get_model(model_name)
        model = Model(num_classes=test_set.num_classes, pretrained=True, freeze=False,
                      keep_head=True, device=device).to(device)
        model.eval()

        if arg.labels or arg.stats:
            # Initialize stats
            stats = {}

            # Iteration
            for step, (video, video_path1, video_path2, label1, label2, video_org) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if model_name == "slowfast":
                        video = [vid.to(device) for vid in video]
                    else:
                        video = video.to(device)

                    # Get predictions
                    preds = model(video)
                    post_act = torch.nn.Softmax(dim=1)
                    preds = post_act(preds)
                    pred_acc = preds.topk(k=5).values[0]
                    pred_classes = preds.topk(k=5).indices[0]
                    pred_class_names = [test_set.kinetics_labels[int(i)] for i in pred_classes]

                    # Update stats
                    stats[video_path1[0].split("/")[-1][:-4] + "_" + video_path2[0].split("/")[-1][:-4]] = \
                        {"label1": test_set.kinetics_labels[label1.item()],
                         "label2": test_set.kinetics_labels[label2.item()],
                         "top_5_acc": pred_acc,
                         "top_5_classes": pred_class_names}

                    if arg.labels:
                        label1_category = "_".join(test_set.kinetics_labels[label1.item()].split(" ")[:3]).replace("(", "").replace(")", "")
                        label2_category = "_".join(test_set.kinetics_labels[label2.item()].split(" ")[:3]).replace("(", "").replace(")", "")
                        prediction_category = "_".join(pred_class_names[0].split(" ")[:3]).replace("(", "").replace(")", "")

                        for method in ["GuidedBackprop", "DeepLift", "Saliency"]:
                            label1_heatmap = get_heatmap(model, video, label1.item(), method)
                            label2_heatmap = get_heatmap(model, video, label2.item(), method)
                            prediction_heatmap = get_heatmap(model, video, pred_classes[0].item(), method)
                            save_heatmap(model_name, label1_heatmap, video_path1, video_path2, method, "label1", label1_category, arg.mix)
                            save_heatmap(model_name, label2_heatmap, video_path1, video_path2, method, "label2", label2_category, arg.mix)
                            save_heatmap(model_name, prediction_heatmap, video_path1, video_path2, method, "prediction", prediction_category, arg.mix)

            if arg.stats:
                print_stats(stats)

        if arg.plot:
            for label in ["prediction", "label2", "label1"]:
                for method in ["GuidedBackprop"]:
                    wandb.init(project=f'Heatmaps Mix {arg.mix}', name=f"{model_name}_{label}_{method}", reinit=True)
                    heatmaps = sorted(glob.glob(f'/export/home/phuber/archive/Heatmap/mix/{arg.mix}/{method}/{model_name}/*{label}*.npy'))
                    plot_heatmaps(heatmaps, test_loader)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)