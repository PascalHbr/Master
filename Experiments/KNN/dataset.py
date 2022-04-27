import numpy as np
import scann


class KNN:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.videos = self.get_videos()
        self.embedding_dim = self.get_embedding_dim()
        self.data_dict, self.category_to_label = self.get_data()
        self.embeddings, self.labels, self.idx_to_category = self.load_embeddings()

        self.searcher = self.create_searcher()

    def get_videos(self):
        if self.data == 'train':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(map(lambda x: f'embeddings/{self.model}/' + x.split(' ')[0][47:-4] + '.npy', f.readlines()))
        elif self.data == 'test':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: f'embeddings/{self.model}/' + x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
            videos = list(sorted(set(videos)))
        videos = list(sorted(set(videos)))

        return videos

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

        print("Found %d videos in %d categories." % (sum(list(map(len, data_dict.values()))),
                                                     len(data_dict)))
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
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.dataset_path = "/export/scratch/compvis/datasets/UCF101/videos"
        self.videos = self.get_videos()
        self.embedding_dim = self.get_embedding_dim()
        self.data_dict, self.category_to_label = self.get_data()
        self.embeddings, self.labels, self.idx_to_category = self.load_embeddings()

    def get_videos(self):
        if self.data == 'train':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/trainlist.txt', 'r') as f:
                videos = list(
                    map(lambda x: f'embeddings/{self.model}/' + x.split(' ')[0][47:-4] + '.npy', f.readlines()))
        elif self.data == 'test':
            with open(f'/export/home/phuber/Master/ImageClassification/splits/UCF101/testlist.txt', 'r') as f:
                videos = list(map(lambda x: f'embeddings/{self.model}/' + x.split(' ')[0][:-1][47:-4] + '.npy', f.readlines()))
        videos = list(sorted(set(videos)))

        return videos

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

        print("Found %d videos in %d categories." % (sum(list(map(len, data_dict.values()))),
                                                     len(data_dict)))
        return data_dict, label_dict

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        # Get video
        embedding = self.embeddings[item]
        category = self.videos[item].split('/')[-1][2:-12]
        label = self.labels[item]

        return embedding, category, label


if __name__ == '__main__':
    knn_classifier = KNN(model="slowfast", data='train')
    queries = Embeddings(model="slowfast", data='test')
    embedding, category, label = queries[2]
    # 3242, 3245, 5912
    neighbors, neighbor_categories = knn_classifier.find_nearest_neighbors(embedding)
    print(category)
    print(neighbor_categories)
    print(neighbors)
    # wandb.init(project='Sample Image')
    # wandb.log({"video": wandb.Video(video[1].permute(1, 0, 2, 3).numpy(), fps=4, format="mp4")})
