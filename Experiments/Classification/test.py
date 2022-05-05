import torch
import torch.nn as nn
from dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


def run_tests(model, device):
    Dataset = get_dataset('ucf')
    test_dataset = Dataset('test', 'slow')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Definde loss function
    criterion = nn.CrossEntropyLoss()

    # Test loop
    total_loss = 0.0
    stats = {}
    total_corrects = 0
    total_counts = 0

    for step, (video, category_idx, label) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            video = video.to(device)
            label = label.to(device)

            # forward
            output = model(video)
            _, pred = torch.max(output, 1)
            loss = criterion(output, label.view(-1))

            # statistics
            total_loss += loss.item() * label.size(0)
            total_corrects += torch.sum(pred == label.data)
            total_counts += 1
            if test_dataset.idx_to_category[category_idx.item()] not in stats:
                stats[test_dataset.idx_to_category[category_idx.item()]] = {"loss": loss.item(),
                                                                            "corrects": torch.sum(
                                                                                pred == label.data).item(),
                                                                            "counts": 1}
            else:
                stats[test_dataset.idx_to_category[category_idx.item()]]["loss"] += loss.item()
                stats[test_dataset.idx_to_category[category_idx.item()]]["corrects"] += torch.sum(
                    pred == label.data).item()
                stats[test_dataset.idx_to_category[category_idx.item()]]["counts"] += 1

    # Print stats
    for category in stats.keys():
        stats[category]["accuracy"] = stats[category]["corrects"] / stats[category]["counts"] * 100
    stats = dict(sorted(stats.items(), key=lambda item: item[1]["accuracy"], reverse=True))
    print(" \n {:<20s} {:<10s} {:<10s} {:<20s}".format("Category", "# Videos", "Loss", "Accuracy (%)"))
    print("-" * 60)
    for key in stats.keys():
        print("{:<20s} {:<10d} {:<10.2f} {:<10.2f}".format(key, stats[key]["counts"],
                                                           stats[key]["loss"] / stats[key]["counts"],
                                                           stats[key]["accuracy"]))
    print("-" * 50)
    print("{:<20s} {:<10d} {:<10.2f} {:<10.2f}".format("Total", total_counts, total_loss / total_counts,
                                                       total_corrects / total_counts * 100))


def predict_archery(model, device):
    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    try:
        urllib.URLopener().retrieve(json_url, json_filename)
    except:
        urllib.request.urlretrieve(json_url, json_filename)

    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second

    url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
    video_path = 'archery.mp4'
    try:
        urllib.URLopener().retrieve(url_link, video_path)
    except:
        urllib.request.urlretrieve(url_link, video_path)

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = inputs.to(device)

    # Pass the input clip through the model
    preds = model(inputs[None, ...])

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))


if __name__ == '__main__':
    # Choose device
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

    # Choose the `slow_r50` model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).to(device)
    model = model.eval()

    predict_archery(model, device)
    # run_tests(model, device)
    # predict_archery(model, device)


