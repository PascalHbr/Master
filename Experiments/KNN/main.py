from dataset import *
from config import *


def main(arg):
    if arg.cfk:
        knn_classifier = KNN(model=arg.model, data='train', name=arg.name, dataset=arg.dataset)
        queries = Embeddings(model=arg.model, data='test', name=arg.name, dataset=arg.dataset)

        stats = {}
        total_corrects = 0
        for (embedding, category, label) in queries:
            neighbors, neighbor_categories = knn_classifier.find_nearest_neighbors(embedding)

            # stats
            total_corrects += int(category == neighbor_categories[0])
            if category not in stats:
                stats[category] = {"corrects": int(category == neighbor_categories[0]),
                                   "counts": 1}
            else:
                stats[category]["corrects"] += int(category == neighbor_categories[0])
                stats[category]["counts"] += 1

        # print stats
        total_accuracy = total_corrects / len(queries) * 100
        for category in stats.keys():
            stats[category]["accuracy"] = stats[category]["corrects"] / stats[category]["counts"] * 100
        stats = dict(sorted(stats.items(), key=lambda item: item[1]["accuracy"], reverse=True))
        print(" \n {:<20s} {:<10s} {:<20s}".format("Category", "# Videos", "Accuracy (%)"))
        print("-" * 50)
        for key in stats.keys():
            print("{:<20s} {:<10d} {:<10.2f}".format(key, stats[key]["counts"], stats[key]["accuracy"]))
        print("-" * 40)
        print("{:<20s} {:<10d} {:<10.2f}".format(f"Total ({arg.model})", len(queries), total_accuracy))

    if arg.plot:
        embeddings = Embeddings(model=arg.model, data='test', name=arg.name, dataset=arg.dataset, subset=False)
        embeddings.plot_embeddings()


if __name__ == '__main__':
    arg = parse_args()
    main(arg)

