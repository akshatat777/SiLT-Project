from data_processing import read_data
import matplotlib.pyplot as plt

train_data, train_label, test_data, test_label = read_data()

for i in range(0, len(train_data), 1000):
    image = train_data[i]
    implot = plt.imshow(image, cmap="gray")
    plt.show()