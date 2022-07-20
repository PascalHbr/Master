from models import *
from tqdm import tqdm

from config import *
from dataset import get_dataset, DataLoader
from utils import *
import glob


def main(arg):
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Datasets
    model_names = ["slow", "x3d", "mvit", "mae"] if arg.model == "all" else [arg.model]
    for model_name in model_names:
        Dataset = get_dataset(arg.dataset)
        test_set = Dataset(mode='test', model=model_name)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Initialize model
        Model = get_model(model_name)
        model = Model(num_classes=test_set.num_classes, pretrained=True, freeze=False,
                      keep_head=True, device=device).to(device)
        model.eval()

        if arg.labels or arg.stats:
            # Make stats dict
            stats = {}

            # Iteration
            for step, (video, video_path, scene_label, action_label, video_org) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if arg.model == 'slowfast':
                        video = [v.to(device) for v in video]
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
                    stats[video_path[0][-6:-4]] = {"scene_label": test_set.kinetics_labels[scene_label.item()],
                                                   "action_label": test_set.kinetics_labels[action_label.item()],
                                                   "top_5_acc": pred_acc,
                                                   "top_5_classes": pred_class_names}

                    if arg.labels:
                        scene_category = "_".join(test_set.kinetics_labels[scene_label.item()].split(" ")[:3]).replace("(", "").replace(")", "")
                        action_category = "_".join(test_set.kinetics_labels[action_label.item()].split(" ")[:3]).replace("(", "").replace(")", "")
                        prediction_category = "_".join(pred_class_names[0].split(" ")[:3]).replace("(", "").replace(")", "")

                        for method in ["GuidedBackprop", "DeepLift", "Saliency"]:
                            scene_heatmap = get_heatmap(model, video, scene_label.item(), method, arg.noise)
                            action_heatmap = get_heatmap(model, video, action_label.item(), method, arg.noise)
                            prediction_heatmap = get_heatmap(model, video, pred_classes[0].item(), method, arg.noise)
                            save_heatmap(model_name, scene_heatmap, video_path, method, "scene", scene_category)
                            save_heatmap(model_name, action_heatmap, video_path, method, "action", action_category)
                            save_heatmap(model_name, prediction_heatmap, video_path, method, "prediction", prediction_category)

            if arg.stats:
                print_stats(stats)

        if arg.plot:
            for label in ["prediction", "action", "scene"]:
                for method in ["GuidedBackprop", "DeepLift", "Saliency"]:
                    wandb.init(project='Heatmap', name=f"{model_name}_{label}_{method}", reinit=True)
                    heatmaps = sorted(glob.glob(f'/export/home/phuber/archive/Heatmap/custom/{method}/{model_name}/*{label}*.npy'))
                    plot_heatmaps(heatmaps, test_loader)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)