import unittest
from getDataNP import get_data

class TestGetData(unittest.TestCase):
    def setUp(self):
        self.train_labels_path = "files/train-labels.idx1-ubyte.gz"
        self.train_images_path = "files/train-images.idx3-ubyte.gz"
        self.test_labels_path = "files/t10k-labels.idx1-ubyte.gz"
        self.test_images_path = "files/t10k-images.idx3-ubyte.gz"

    def test_get_data_valid_count_training_files(self):
        count = 10
        labels, images = get_data(count, self.train_labels_path, self.train_images_path)
        self.assertEqual(labels.shape, (count,))
        self.assertEqual(images.shape[0], count)
        self.assertEqual(images.shape[1:], (28, 28))  # Assuming MNIST image size
    
    def test_get_data_valid_count_test_files(self):
        count = 10
        labels, images = get_data(count, self.test_labels_path, self.test_images_path)
        self.assertEqual(labels.shape, (count,))
        self.assertEqual(images.shape[0], count)
        self.assertEqual(images.shape[1:], (28, 28))  # Assuming MNIST image size

    def test_get_data_zero_count(self):
        count = 0
        labels, images = get_data(count, self.train_labels_path, self.train_images_path)
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(images.shape, (0, 28, 28))

    def test_get_data_exceeding_count(self):
        count = 70000  # Exceeds MNIST training set size
        labels, images = get_data(count, self.train_labels_path, self.train_images_path)
        self.assertEqual(labels.shape, (60000,))
        self.assertEqual(images.shape[0], 60000)
        self.assertEqual(images.shape[1:], (28, 28))

    def test_get_data_negative_count(self):
        count = -5
        labels, images = get_data(count, self.train_labels_path, self.train_images_path)
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(images.shape, (0, 28, 28))

class TestGetDataFiles(unittest.TestCase):
    def setUp(self):
        self.train_labels_path = "files/train-labels-idx1-ubyte.gz"
        self.train_images_path = "files/train-images-idx3-ubyte.gz"

    def test_training_images(self):
        labels, images = get_data(1, self.train_labels_path, self.train_images_path)
        print(images)
        self.assertTrue(True)


class TestAi(unittest.TestCase):
    def test_placeholder(self):
        self.assertTrue(True)

# class TestFail(unittest.TestCase):
#     def test_fail(self):
#         self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()