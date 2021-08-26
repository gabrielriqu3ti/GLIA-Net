import unittest


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.patch_size = (96, 96, 96)
        self.img_shape_list = [(138, 512, 512),
                               (138, 512, 512),
                               (138, 512, 512),
                               (138, 512, 512),
                               (138, 512, 512),
                               (138, 512, 512),
                               (140, 560, 560),
                               ]
        self.start_list = [(0, 0, 0),
                           (0, 210, 161),
                           (0, 193, 190),
                           (0, 369, 123),
                           (29, 369, 8),
                           (40, 371, 16),
                           (40, 420, 371),
                           ]

        self.end_list = [tuple(start_i + 96 for start_i in start) for start in self.start_list]
        self.start_ext = [tuple(max(start_i - 48, 0) for start_i in start) for start in self.start_list]
        self.end_ext = [tuple(min(end_i + 48, shape_i) for (end_i, shape_i) in zip(end, shape))
                        for (end, shape) in zip(self.end_list, self.img_shape_list)]
        self.start_border = [tuple(max(48 - start_i, 0) for start_i in start) for start in self.start_list]
        self.end_border = [tuple(border_i + end_ext_i - start_ext_i for (border_i, start_ext_i, end_ext_i)
                                 in zip(start_border, start_ext, end_ext)) for (start_border, start_ext, end_ext)
                           in zip(self.start_border, self.start_ext, self.end_ext)]

    def test_patch_indexing(self):
        for i in range(len(self.img_shape_list)):
            input_img = [self.end_ext[i][j] - self.start_ext[i][j] for j in range(3)]
            output_patch = [self.end_border[i][j] - self.start_border[i][j] for j in range(3)]
            self.assertEqual(input_img, output_patch)


if __name__ == '__main__':
    unittest.main()
