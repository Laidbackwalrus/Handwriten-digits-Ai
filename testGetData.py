import unittest
from getDataNP import get_data
from getDataNP import display_images

class getDateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train_images_file = "files/train-images.idx3-ubyte.gz"
        self.train_labels_file = "files/train-labels.idx1-ubyte.gz"
        self.test_images_file = "files/t10k-images.idx3-ubyte.gz"
        self.test_labels_file = "files/t10k-labels.idx1-ubyte.gz"

    def test_sanity_train_files(self):
        try:
            labels, images = get_data(10, self.train_labels_file, self.train_images_file)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"get_data raised an exception: {e}")

    def test_sanity_test_files(self):
        try:
            labels, images = get_data(10, self.test_labels_file, self.test_images_file)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"get_data raised an exception: {e}")

    def test_output_shapes(self):
        labels, images = get_data(10, self.train_labels_file, self.train_images_file)
        self.assertEqual(labels.shape, (10,))
        self.assertEqual(images.shape[0], 10)
        self.assertEqual(images.shape[1:], (28, 28))

        labels, images = get_data(10, self.test_labels_file, self.test_images_file)
        self.assertEqual(labels.shape, (10,))
        self.assertEqual(images.shape[0], 10)
        self.assertEqual(images.shape[1:], (28, 28))

    def test_count_large(self):
        labels, images = get_data(70000, self.train_labels_file, self.train_images_file)
        self.assertEqual(labels.shape, (60000,))
        self.assertEqual(images.shape[0], 60000)
        self.assertEqual(images.shape[1:], (28, 28))

    @unittest.skipIf(not globals().get('manualcheck', True), "Manual check disabled")
    def test_by_observation(self):
        labels, images = get_data(10, self.train_labels_file, self.train_images_file)
        print (labels)
        display_images(images)
        user_input = input("Did the displayed image match the label? (y/n): ")
        self.assertTrue(user_input.lower() == 'y', "User indicated mismatch between image and label")
        
        labels, images = get_data(1, self.test_labels_file, self.test_images_file)
        print (labels)
        display_images(images)
        user_input = input("Did the displayed image match the label? (y/n): ")
        self.assertTrue(user_input.lower() == 'y', "User indicated mismatch between image and label")



if __name__ == '__main__':
    manualcheck = True  # Set to True to enable manual checks
    unittest.main()