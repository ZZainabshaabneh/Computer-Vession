
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_moments(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    moments = []
    for channel in range(3):
        channel_data = hsv[:, :, channel]
        moments.extend([np.mean(channel_data), np.std(channel_data), np.mean(np.power(channel_data, 3))])
    return np.array(moments)

def calculate_histogram(image, bins):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_r, bins_r = np.histogram(hsv[:,:,0], bins, range=(0, 256), density=True)
    hist_g, bins_g = np.histogram(hsv[:,:,1], bins, range=(0, 256), density=True)
    hist_b, bins_b = np.histogram(hsv[:,:,2], bins, range=(0, 256), density=True)
    hist_features = np.concatenate((hist_r, hist_g, hist_b))
    return hist_features

def calculate_similarity(query_features, image_features):
    # Use Euclidean distance as a measure of similarity
    return np.linalg.norm(query_features - image_features)

def search_similar_images(query_image_path, database_path, feature_type):
    query_image = cv2.imread(query_image_path)
    if feature_type == "moment":
        query_features = calculate_moments(query_image)
    elif feature_type == "histo":
        query_features = calculate_histogram(query_image, bins=256)

    similar_images = []
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(database_path, filename)
            database_image = cv2.imread(image_path)
            if feature_type == "moment":
                image_features = calculate_moments(database_image)
            elif feature_type == "histo":
                image_features = calculate_histogram(database_image, bins=256)
            similarity = calculate_similarity(query_features, image_features)
            similar_images.append((filename, similarity))

    similar_images.sort(key=lambda x: x[1])

    for rank, (filename, similarity) in enumerate(similar_images, start=1):
        similar_images[rank - 1] = (rank, filename, similarity)

    return similar_images

def plot_images(query_image, similar_images, database_path, feature_type):
    plt.figure(figsize=(12, 8))
    plt.suptitle(feature_type, fontsize=16)

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title('Query Image')
    plt.axis('off')

    for i in range(5):
        rank, filename, similarity = similar_images[i]
        image_path = os.path.join(database_path, filename)
        similar_image = cv2.imread(image_path)

        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Rank {rank}\nSimilarity: {similarity:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    database_path = "/content/content/Images"
    query_image_path = "/content/content/query/8.jpg"
    similar_images_moments = search_similar_images(query_image_path, database_path, "moment")
    similar_images_histogram = search_similar_images(query_image_path, database_path, "histo")

    query_image = cv2.imread(query_image_path)
    plot_images(query_image, similar_images_moments, database_path, "Color Moment Features")
    plot_images(query_image, similar_images_histogram, database_path, "Color Histogram Features")
