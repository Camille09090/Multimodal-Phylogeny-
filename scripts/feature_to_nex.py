import os

import numpy as np


OUTPUT_DIR = "./feature"
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.npy")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.npy")
INDEX_TO_LABEL_PATH = os.path.join(OUTPUT_DIR, "index_to_label.npy")
NEXUS_PATH = os.path.join(OUTPUT_DIR, "feature.nex")


def load_inputs():
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    index_to_label = np.load(INDEX_TO_LABEL_PATH, allow_pickle=True).item()
    return features, labels, index_to_label


def build_label_names(labels, index_to_label):
    return np.asarray([index_to_label[int(i)] for i in labels], dtype=object)


def compute_class_mean_features(features, label_names):
    unique_labels = sorted(set(label_names.tolist()))
    mean_features = []

    for label in unique_labels:
        class_features = features[label_names == label]
        mean_features.append(class_features.mean(axis=0))

    return unique_labels, np.asarray(mean_features, dtype=np.float32)


def standardize_features(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (features - mean) / std


def format_taxon_name(name):
    return str(name).replace(" ", "_")


def save_nexus(nexus_path, labels_sorted, final_features):
    n_taxa = len(labels_sorted)
    n_char = final_features.shape[1]

    with open(nexus_path, "w", encoding="utf-8") as handle:
        handle.write("#NEXUS\n")
        handle.write("Begin data;\n")
        handle.write("Dimensions ntax=%d nchar=%d;\n" % (n_taxa, n_char))
        handle.write("Format datatype=continuous missing=?;\n")
        handle.write("Matrix\n")

        for i, label in enumerate(labels_sorted):
            taxon_name = format_taxon_name(label)
            line = "%-20s %s" % (
                taxon_name,
                " ".join("%.6f" % x for x in final_features[i]),
            )
            handle.write(line + "\n")

        handle.write(";\n")
        handle.write("End;\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features, labels, index_to_label = load_inputs()
    label_names = build_label_names(labels, index_to_label)
    labels_sorted, mean_features = compute_class_mean_features(features, label_names)
    final_features = standardize_features(mean_features)

    save_nexus(NEXUS_PATH, labels_sorted, final_features)

    print("\n===================================")
    print("Dual-space BERT CLS feature to NEXUS done")
    print("features shape:", features.shape)
    print("class mean feature shape:", mean_features.shape)
    print("n_taxa:", len(labels_sorted))
    print("n_char:", final_features.shape[1])
    print("nexus_path:", NEXUS_PATH)


if __name__ == "__main__":
    main()
