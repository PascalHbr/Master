import numpy as np
import scann
from sklearn.manifold import TSNE
import time
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from urllib import request


class KNN:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')

        self.all_videos, self.all_embeddings = self.get_all_videos()
        print(len(self.all_videos))
        print(len(self.all_embeddings))
        self.categories, self.label_dict = self.get_classification_categories()
        self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        self.embedding_dim = self.get_embedding_dim()
        self.data_dict, self.category_to_label = self.get_data()
        self.embeddings, self.labels, self.idx_to_category = self.load_embeddings()

        self.searcher = self.create_searcher()

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_all_videos(self):
        if self.data == 'train':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.model}/' + x.split(' ')[0][47:-4] + '.npy', f.readlines()))
        elif self.data == 'test':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0][:-1], f.readlines()))
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.model}/' + x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
        videos = list(sorted(set(videos)))
        embeddings = list(sorted(set(embeddings)))

        return videos, embeddings

    def get_classification_categories(self):
        categories = {}
        for video, embedding in zip(self.all_videos, self.all_embeddings):
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            if video in self.all_videos:
                categories[category].append(embedding)

        # Choose selected categories from labels.txt
        valid_categories = []
        if self.model in ["vimpac", "mae"]:
            selected_categories = categories
        else:
            for i, (category, sequences) in enumerate(categories.items()):
                kinetics_id = self.ucf_labels[i]
                if kinetics_id >= 0:
                    valid_categories.append(category)
            selected_categories = {valid_category: categories[valid_category] for valid_category in valid_categories}

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            if category in selected_categories:
                label_dict[category] = self.ucf_labels[i]

        print("Found %d videos in %d categories." % (sum(list(map(len, selected_categories.values()))),
                                                     len(selected_categories)))
        return selected_categories, label_dict

    def get_embedding_dim(self):
        if self.model == 'slow':
            return 2048
        elif self.model == 'slowfast':
            return 2304
        elif self.model == 'x3d':
            return 2048
        elif self.model == 'mvit':
            return 768
        elif self.model == 'vimpac':
            return 512
        elif self.model == 'mae':
            return 768

    def load_embeddings(self):
        embeddings = np.zeros((len(self.videos), self.embedding_dim), dtype=np.float32)
        labels = np.zeros((len(self.videos), 1), dtype=np.int8)
        idx_to_category = {}
        for i, video in enumerate(self.videos):
            embeddings[i, :] = np.load(video)
            labels[i, :] = self.category_to_label[video.split('/')[-1][2:-12]]
            idx_to_category[i] = video.split('/')[-1][2:-12]
        return embeddings, labels, idx_to_category

    def get_data(self):
        data_dict = {}
        for video in self.videos:
            category = video.split('/')[-1][2:-12]
            if category not in data_dict:
                data_dict[category] = []
            data_dict[category].append(video)

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(data_dict):
            label_dict[category] = i

        return data_dict, label_dict

    def print_summary(self):
        for category, sequences in self.data_dict.items():
            summary = ", ".join(sequences[:1])
            print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))

    def create_searcher(self):
        normalized_dataset = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
            num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(100).build()

        return searcher

    def find_nearest_neighbors(self, embedding, k=10):
        neighbors, distances = self.searcher.search(embedding, final_num_neighbors=k)
        neighbors = sorted(x for _, x in sorted(zip(distances, neighbors)))
        neighbor_categories = [self.idx_to_category[neighbor] for neighbor in neighbors]

        return neighbors, neighbor_categories

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        embedding = self.embeddings[item]
        category = self.videos[item].split('/')[-1][2:-12]
        label = self.labels[item]

        return embedding, category, label


class Embeddings:
    def __init__(self, model, data, subset=False):
        self.model = model
        self.data = data

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')

        self.all_videos, self.all_embeddings = self.get_all_videos()
        self.categories, self.label_dict = self.get_classification_categories()
        self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        self.embedding_dim = self.get_embedding_dim()
        self.data_dict, self.category_to_label = self.get_data()
        self.embeddings, self.labels, self.idx_to_category = self.load_embeddings()
        if subset:
            subset_length = self.embeddings.shape[0] // 10
            self.embeddings, self.labels = self.embeddings[:subset_length], self.labels[:subset_length]

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_all_videos(self):
        if self.data == 'train':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.model}/' + x.split(' ')[0][47:-4] + '.npy', f.readlines()))
        elif self.data == 'test':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: x.split(' ')[0][:-1], f.readlines()))
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.model}/' + x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
        videos = list(sorted(set(videos)))
        embeddings = list(sorted(set(embeddings)))

        return videos, embeddings

    def get_classification_categories(self):
        categories = {}
        for video, embedding in zip(self.all_videos, self.all_embeddings):
            category = video.split('/')[-1][2:-12]
            if category not in categories:
                categories[category] = []
            if video in self.all_videos:
                categories[category].append(embedding)

        # Choose selected categories from labels.txt
        valid_categories = []
        if self.model in ["vimpac", "mae"]:
            selected_categories = categories
        else:
            for i, (category, sequences) in enumerate(categories.items()):
                kinetics_id = self.ucf_labels[i]
                if kinetics_id >= 0:
                    valid_categories.append(category)
            selected_categories = {valid_category: categories[valid_category] for valid_category in valid_categories}

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            if category in selected_categories:
                label_dict[category] = self.ucf_labels[i]

        print("Found %d videos in %d categories." % (sum(list(map(len, selected_categories.values()))),
                                                     len(selected_categories)))
        return selected_categories, label_dict

    def get_embedding_dim(self):
        if self.model == 'slow':
            return 2048
        elif self.model == 'slowfast':
            return 2304
        elif self.model == 'x3d':
            return 2048
        elif self.model == 'mvit':
            return 768
        elif self.model == 'vimpac':
            return 512
        elif self.model == 'mae':
            return 768

    def load_embeddings(self):
        embeddings = np.zeros((len(self.videos), self.embedding_dim), dtype=np.float32)
        labels = np.zeros((len(self.videos), 1), dtype=np.int8)
        idx_to_category = {}
        for i, video in enumerate(self.videos):
            embeddings[i, :] = np.load(video)
            labels[i, :] = self.category_to_label[video.split('/')[-1][2:-12]]
            idx_to_category[i] = video.split('/')[-1][2:-12]
        return embeddings, labels, idx_to_category

    def get_data(self):
        data_dict = {}
        for video in self.videos:
            category = video.split('/')[-1][2:-12]
            if category not in data_dict:
                data_dict[category] = []
            data_dict[category].append(video)

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(data_dict):
            label_dict[category] = i

        return data_dict, label_dict

    def plot_embeddings(self):
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(self.embeddings)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        # log scatterplot on wandb
        wandb.init(project='t-sne', name=self.model)
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        categories = [self.idx_to_category[i] for i in range(self.labels.shape[0])]
        df = pd.DataFrame(tsne_results, categories, columns=["x", "y"])
        df["Category"] = categories
        sns.scatterplot(data=df, x="x", y="y", hue="Category")
        # plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        wandb.log({f"t-sne of {self.model}": wandb.Image(fig)})

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        embedding = self.embeddings[item]
        category = self.videos[item].split('/')[-1][2:-12]
        label = self.labels[item]

        return embedding, category, label


if __name__ == '__main__':
    knn_classifier = KNN(model="slowfast", data='test')
    # embeddings = Embeddings(model="mvit", data='test', subset=True)
    # embeddings.plot_embeddings()
