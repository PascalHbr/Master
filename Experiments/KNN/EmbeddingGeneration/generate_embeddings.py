import torch.utils.data
from config import *
from dataset import *
from tqdm import tqdm


def main(arg):
    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')

    # Setup Dataloader and model
    Dataset = get_dataset(arg.dataset)
    train_dataset = Dataset('train', arg.model)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True,
                              generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)
    test_dataset = Dataset('test', arg.model)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True,
                             generator=torch.Generator().manual_seed(42), worker_init_fn=seed_worker)

    # Initialize model
    Model = get_model(arg.model)
    model = Model(device=device).to(device)
    model.eval()

    # Set random seeds
    set_random_seed()

    # Train loop
    # for step, (videos, video_paths) in enumerate(tqdm(train_loader)):
    #     with torch.no_grad():
    #         if arg.model == 'slowfast':
    #             videos = [video.to(device) for video in videos]
    #         else:
    #             videos = videos.to(device)
    #
    #         # create and save embeddings
    #         embeddings = model(videos)
    #         save_embedding(embeddings, video_paths, arg.dataset, arg.name)

    # Test loop
    for step, (videos, video_paths) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if arg.model == 'slowfast':
                videos = [video.to(device) for video in videos]
            else:
                videos = videos.to(device)

            # create and save embeddings
            embeddings = model(videos)
            save_embedding(embeddings, video_paths, arg.dataset, arg.name)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)