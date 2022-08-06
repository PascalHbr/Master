import numpy as np
import scann
from sklearn.manifold import TSNE
import time
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from urllib import request
import json
import csv


class KNN:
    def __init__(self, model, data, name, dataset):
        self.model = model
        self.data = data
        self.name = name
        self.dataset = dataset

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')
        if self.dataset == "ssv2":
            self.data = "validation"
            self.annotations = self.get_annotations()
            self.ssv2_videos, self.ssv2_labels = self.get_ssv2_videos()
        elif self.dataset == "kinetics":
            self.base_path = '/export/data/compvis/kinetics/'
            self.data = "eval"
            self.annotations = self.get_kinetics_annotations()
            self.kinetics_videos, self.kinetics_labels = self.get_kinetics_videos()
        self.all_videos, self.all_embeddings = self.get_all_videos()

        if self.dataset == "ucf":
            self.categories, self.label_dict = self.get_classification_categories()
            self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        elif self.dataset == "ssv2":
            self.categories, self.label_dict = self.get_ssv2_categories()
            self.videos = self.all_embeddings
        elif self.dataset == "kinetics":
            self.categories, self.label_dict = self.get_kinetics_categories()
            self.videos = self.all_embeddings
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

    def get_annotations(self):
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/' + self.data + '.json'
        with open(label_path, 'r') as json_file:
            annotations = json.load(json_file)

        return annotations

    def get_kinetics_annotations(self):
        if self.data == 'train':
            annotations = pd.read_csv(self.base_path + 'train.csv')
            with open('/export/home/phuber/archive/valid_train_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]
        else:
            annotations = pd.read_csv(self.base_path + 'validate.csv')
            with open('/export/home/phuber/archive/valid_test_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]

        return annotations.sort_values(by=['label']).reset_index(drop=True)

    def get_ssv2_videos(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        videos = [video_base_path + data['id'] + '.webm' for data in self.annotations]
        labels = [data['template'].replace('[', '').replace(']', '') for data in self.annotations]

        return videos, labels

    def get_kinetics_videos(self):
        videos = [self.base_path + self.data + '/' + self.annotations['label'][i] + '/' +
                  self.annotations['youtube_id'][i] for i in range(len(self.annotations))]
        videos = [vid.replace(' ', '_') for vid in videos]
        labels = self.annotations['label']

        return videos, labels

    def get_ssv2_categories(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        categories = {}
        for data in self.annotations:
            category = data['template'].replace('[', '').replace(']', '')
            if category not in categories:
                categories[category] = []
            categories[category].append(video_base_path + data['id'] + '.webm')

        # Make dictionary that gives kinetics label for category
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/labels.json'
        with open(label_path, 'r') as json_file:
            label_dict = json.load(json_file)

        print("Found %d videos in %d categories." % (sum(list(map(len, categories.values()))),
                                                     len(categories)))
        return categories, label_dict

    def get_kinetics_categories(self):
        categories = {}
        for index, row in self.annotations.iterrows():
            category = row['label']
            if category not in categories:
                categories[category] = []
            categories[category].append(row['youtube_id'])

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            label_dict[category] = i

        print("Found %d videos in %d categories." % (sum(list(map(len, categories.values()))),
                                                     len(categories)))
        return categories, label_dict

    def get_all_videos(self):
        if self.dataset == "ucf":
            if self.data == 'train':
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                    videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                    embeddings = list(
                        map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' +
                                      x.split(' ')[0][47:-4] + '.npy', f.readlines()))
            elif self.data == 'test':
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                    videos = list(map(lambda x: x.split(' ')[0][:-1], f.readlines()))
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                    embeddings = list(
                        map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' +
                                      x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
            videos = list(sorted(set(videos)))
            embeddings = list(sorted(set(embeddings)))

        elif self.dataset == "ssv2":
            videos = self.ssv2_videos
            embeddings = list(
                map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' + x[54:-5] + '.npy',
                    self.ssv2_videos))

        elif self.dataset == "kinetics":
            videos = self.kinetics_videos
            embeddings = list(
                map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' + x[35:] + '.npy',
                    self.kinetics_videos))

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
            if self.dataset == "ucf":
                labels[i, :] = self.category_to_label[video.split('/')[-1][2:-12]]
                idx_to_category[i] = video.split('/')[-1][2:-12]
            elif self.dataset == "ssv2":
                labels[i, :] = self.category_to_label[self.ssv2_labels[i]]
                idx_to_category[i] = self.ssv2_labels[i]
            elif self.dataset == "kinetics":
                labels[i, :] = self.category_to_label[self.kinetics_labels[i]]
                idx_to_category[i] = self.kinetics_labels[i]

        return embeddings, labels, idx_to_category

    def get_data(self):
        if self.dataset == "ucf":
            data_dict = {}
            for video in self.videos:
                category = video.split('/')[-1][2:-12]
                if category not in data_dict:
                    data_dict[category] = []
                data_dict[category].append(video)
        elif self.dataset == "ssv2":
            data_dict = {}
            for i, video in enumerate(self.videos):
                category = self.ssv2_labels[i]
                if category not in data_dict:
                    data_dict[category] = []
                data_dict[category].append(video)
        elif self.dataset == "kinetics":
            data_dict = {}
            for i, video in enumerate(self.videos):
                category = self.kinetics_labels[i]
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
        if self.dataset == "ucf":
            category = self.videos[item].split('/')[-1][2:-12]
        elif self.dataset == "ssv2":
            category = self.ssv2_labels[item]
        elif self.dataset == "kinetics":
            category = self.kinetics_labels[item]
        label = self.labels[item]

        return embedding, category, label


class Embeddings:
    def __init__(self, model, data, name, dataset, subset=False):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.name = name

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.kinetics_labels = self.read_kinetics_labels()
        self.ucf_labels = self.read_ucf_labels('/export/home/phuber/Master/I3D_augmentations/labels.csv')
        if self.dataset == "ssv2":
            self.data = "validation"
            self.annotations = self.get_annotations()
            self.ssv2_videos, self.ssv2_labels = self.get_ssv2_videos()
        elif self.dataset == "kinetics":
            self.base_path = '/export/data/compvis/kinetics/'
            self.data = "eval"
            self.annotations = self.get_kinetics_annotations()
            self.kinetics_videos, self.kinetics_labels = self.get_kinetics_videos()
        self.all_videos, self.all_embeddings = self.get_all_videos()

        if self.dataset == "ucf":
            self.categories, self.label_dict = self.get_classification_categories()
            self.videos = np.array([item for sublist in self.categories.values() for item in sublist])
        elif self.dataset == "ssv2":
            self.categories, self.label_dict = self.get_ssv2_categories()
            self.videos = self.all_embeddings
        elif self.dataset == "kinetics":
            self.categories, self.label_dict = self.get_kinetics_categories()
            self.videos = self.all_embeddings
        self.embedding_dim = self.get_embedding_dim()
        self.data_dict, self.category_to_label = self.get_data()
        self.embeddings, self.labels, self.idx_to_category = self.load_embeddings()

        if subset:
            subset_length = self.embeddings.shape[0] // 8
            self.embeddings, self.labels = self.embeddings[:subset_length], self.labels[:subset_length]

    def read_kinetics_labels(self):
        KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(KINETICS_URL) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def read_ucf_labels(self, path_labels):
        labels = pd.read_csv(path_labels)["Corresponding Kinetics Labels"].to_list()
        return labels

    def get_annotations(self):
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/' + self.data + '.json'
        with open(label_path, 'r') as json_file:
            annotations = json.load(json_file)

        return annotations

    def get_kinetics_annotations(self):
        if self.data == 'train':
            annotations = pd.read_csv(self.base_path + 'train.csv')
            with open('/export/home/phuber/archive/valid_train_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]
        else:
            annotations = pd.read_csv(self.base_path + 'validate.csv')
            with open('/export/home/phuber/archive/valid_test_ids.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                valid_ids = [int(row[0]) for row in reader if row]
            annotations = annotations[annotations.index.isin(valid_ids)]

        return annotations.sort_values(by=['label']).reset_index(drop=True)

    def get_ssv2_videos(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        videos = [video_base_path + data['id'] + '.webm' for data in self.annotations]
        labels = [data['template'].replace('[', '').replace(']', '') for data in self.annotations]

        return videos, labels

    def get_kinetics_videos(self):
        videos = [self.base_path + self.data + '/' + self.annotations['label'][i] + '/' +
                  self.annotations['youtube_id'][i] for i in range(len(self.annotations))]
        videos = [vid.replace(' ', '_') for vid in videos]
        labels = self.annotations['label']

        return videos, labels

    def get_ssv2_categories(self):
        video_base_path = '/export/scratch/compvis/datasets/somethingsomethingv2/20bn-something-something-v2/'
        categories = {}
        for data in self.annotations:
            category = data['template'].replace('[', '').replace(']', '')
            if category not in categories:
                categories[category] = []
            categories[category].append(video_base_path + data['id'] + '.webm')

        # Make dictionary that gives kinetics label for category
        label_path = '/export/scratch/compvis/datasets/somethingsomethingv2/labels/labels.json'
        with open(label_path, 'r') as json_file:
            label_dict = json.load(json_file)

        print("Found %d videos in %d categories." % (sum(list(map(len, categories.values()))),
                                                     len(categories)))
        return categories, label_dict

    def get_kinetics_categories(self):
        categories = {}
        for index, row in self.annotations.iterrows():
            category = row['label']
            if category not in categories:
                categories[category] = []
            categories[category].append(row['youtube_id'])

        # Make dictionary that gives kinetics label for category
        label_dict = {}
        for i, category in enumerate(categories):
            label_dict[category] = i

        print("Found %d videos in %d categories." % (sum(list(map(len, categories.values()))),
                                                     len(categories)))
        return categories, label_dict

    def get_all_videos(self):
        if self.dataset == "ucf":
            if self.data == 'train':
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                    videos = list(map(lambda x: x.split(' ')[0], f.readlines()))
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                    embeddings = list(
                        map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' +
                                      x.split(' ')[0][47:-4] + '.npy', f.readlines()))
            elif self.data == 'test':
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                    videos = list(map(lambda x: x.split(' ')[0][:-1], f.readlines()))
                with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                    embeddings = list(
                        map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' +
                                      x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
            videos = list(sorted(set(videos)))
            embeddings = list(sorted(set(embeddings)))

        elif self.dataset == "ssv2":
            videos = self.ssv2_videos
            embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' + x[54:-5] + '.npy', self.ssv2_videos))

        elif self.dataset == "kinetics":
            videos = self.kinetics_videos
            embeddings = list(map(lambda x: f'/export/home/phuber/archive/embeddings/{self.dataset}/{self.name}/' + x[35:] + '.npy', self.kinetics_videos))

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
            if self.dataset == "ucf":
                labels[i, :] = self.category_to_label[video.split('/')[-1][2:-12]]
                idx_to_category[i] = video.split('/')[-1][2:-12]
            elif self.dataset == "ssv2":
                labels[i, :] = self.category_to_label[self.ssv2_labels[i]]
                idx_to_category[i] = self.ssv2_labels[i]
            elif self.dataset == "kinetics":
                labels[i, :] = self.category_to_label[self.kinetics_labels[i]]
                idx_to_category[i] = self.kinetics_labels[i]

        return embeddings, labels, idx_to_category

    def get_data(self):
        if self.dataset == "ucf":
            data_dict = {}
            for video in self.videos:
                category = video.split('/')[-1][2:-12]
                if category not in data_dict:
                    data_dict[category] = []
                data_dict[category].append(video)
        elif self.dataset == "ssv2":
            data_dict = {}
            for i, video in enumerate(self.videos):
                category = self.ssv2_labels[i]
                if category not in data_dict:
                    data_dict[category] = []
                data_dict[category].append(video)
        elif self.dataset == "kinetics":
            data_dict = {}
            for i, video in enumerate(self.videos):
                category = self.kinetics_labels[i]
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
        wandb.init(project=f't-sne {self.dataset}', name=self.name)
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        categories = [self.idx_to_category[i] for i in range(self.labels.shape[0])]
        df = pd.DataFrame(tsne_results, categories, columns=["x", "y"])
        df["Category"] = categories
        sns.scatterplot(data=df, x="x", y="y", hue="Category", legend=False)
        # plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        wandb.log({f"t-sne of {self.model}": wandb.Image(fig)})

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        embedding = self.embeddings[item]
        if self.dataset == "ucf":
            category = self.videos[item].split('/')[-1][2:-12]
        elif self.dataset == "ssv2":
            category = self.ssv2_labels[item]
        elif self.dataset == "kinetics":
            category = self.kinetics_labels[item]
        label = self.labels[item]

        return embedding, category, label


if __name__ == '__main__':
    # knn_classifier = KNN(model="slow", data='test', name="slow", dataset="ssv2")
    embeddings = KNN(model="slow", data='test', name="slow", dataset="kinetics")
    embedding, category, label = embeddings[0]
    print(category)
    print(label)
    # embeddings = Embeddings(model="mvit", data='test', subset=True)
    # embeddings.plot_embeddings()
