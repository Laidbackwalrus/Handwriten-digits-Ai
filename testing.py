import unittest
import numpy as np
from aiV3 import myAI
from aiV3 import label_to_vector
from aiV3 import batch_label_to_vector
from aiV3 import sigmoid

from getDataNP import get_data

class TestAi(unittest.TestCase):
    def setUp(self) -> None:
        self.ai = myAI([784, 50, 30, 10], 0.01)
    
    def test_initialization(self):
        """Structure tests"""
        self.assertEqual(len(self.ai.layers), 4)
        self.assertEqual(self.ai.layers[0].num_nodes, 784)
        self.assertEqual(self.ai.layers[1].num_nodes, 50)
        self.assertEqual(self.ai.layers[2].num_nodes, 30)
        self.assertEqual(self.ai.layers[3].num_nodes, 10)

    def test_predict_shape(self):
        """Predict output shape test"""
        input_data = np.random.rand(784,)
        output = self.ai.predict(input_data)
        self.assertEqual(output.shape, (10,))

    def test_sigmoid(self):
        """Sigmoid function test"""
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1000), 1.0)
        self.assertAlmostEqual(sigmoid(-1000), 0.0)

    def test_batch_to_vector(self):
        """Batch label to vector conversion test"""
        num = 10
        labels = np.array(range(num))
        vectors = batch_label_to_vector(labels)
        self.assertEqual(vectors.shape, (num, 10))
        for i in labels:
            np.testing.assert_array_equal(vectors[i], label_to_vector(i))


    def test_label_to_vector(self):
        """Single label to vector conversion test"""
        vector = label_to_vector(5)
        self.assertEqual(vector.shape, (10,))
        self.assertEqual(vector.dtype, np.float32)
        expected = np.array([0,0,0,0,0,1,0,0,0,0])
        np.testing.assert_array_equal(expected, vector.flatten())

    def test_numpy_rows_and_columns(self):
        arr = np.arange(12).reshape(3, 4)
        self.assertEqual(arr.shape, (3, 4)) # 3 list of 4 items!!!

    def test_output_shapes(self):
        """Check shapes of weights, biases, and values in each layer"""
        for i in range(1, len(self.ai.layers)):
            prev_layer = self.ai.layers[i - 1]
            curr_layer = self.ai.layers[i]
            self.assertEqual(curr_layer.weights.shape, (curr_layer.num_nodes, prev_layer.num_nodes))
            self.assertEqual(curr_layer.biases.shape, (curr_layer.num_nodes, ))
            self.assertEqual(curr_layer.values.shape, (curr_layer.num_nodes, ))

    def test_cost_function(self):
        """Cost function test"""
        labels, images = get_data(3, "files/t10k-labels.idx1-ubyte.gz", "files/t10k-images.idx3-ubyte.gz")
        vectorised_labels = batch_label_to_vector(labels)

        for (image, v_label) in zip(images, vectorised_labels):
            prediction = self.ai.predict(image)
            cost = self.ai.cost_function(prediction, v_label)

            # Cost between 0, 2
            self.assertGreaterEqual(cost, 0, f"Cost {cost} should be >= 0")
            self.assertLessEqual(cost, 2, f"Cost {cost} should be <= 2")

    def test_sanity_matrix_multiplication(self):
        m, n = 1, 2
        n, p = 2, 3

        a = np.zeros((m, n))
        b = np.zeros((n, p))

        a_p = np.zeros((n, m))
        b_p = np.zeros((p, n))

        c = a @ b
        self.assertEqual(c.shape, (1, 3)) # 
        with self.assertRaises(ValueError, msg="m != p"):
            C_invalid = a_p * b_p
            


if __name__ == '__main__':
    unittest.main()