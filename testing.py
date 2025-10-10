import unittest
import numpy as np
from getDataNP import get_data
from getDataNP import display_images
from aiV3 import myAI

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

    

if __name__ == '__main__':
    
    unittest.main()