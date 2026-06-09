from util_meter import AverageMeter, accuracy
import torch
import unittest
from util_meter import AverageMeter

# class TestAverageMeter(unittest.TestCase):
#     def test_reset(self):
#         meter = AverageMeter('test')
#         meter.update(5)
#         meter.reset()
#         self.assertEqual(meter.val, 0)
#         self.assertEqual(meter.avg, 0)
#         self.assertEqual(meter.sum, 0)
#         self.assertEqual(meter.count, 0)

#     def test_update(self):
#         meter = AverageMeter('test')
#         meter.update(5)
#         self.assertEqual(meter.val, 5)
#         self.assertEqual(meter.avg, 5)
#         self.assertEqual(meter.sum, 5)
#         self.assertEqual(meter.count, 1)

#         meter.update(10, 2)
#         self.assertEqual(meter.val, 10)
#         self.assertEqual(meter.avg, 7.5)
#         self.assertEqual(meter.sum, 25)
#         self.assertEqual(meter.count, 3)

#     def test_str(self):
#         meter = AverageMeter('test')
#         meter.update(5)
#         self.assertEqual(str(meter), 'test5.000000(5.000000)')

#         meter.update(10, 2)
#         self.assertEqual(str(meter), 'test10.000000(7.500000)')


class TestAccuracy(unittest.TestCase):
    def test_accuracy(self):
        output = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        target = torch.tensor([2, 0])
        topk = (1, 3)
        result = accuracy(output, target, topk)
        self.assertEqual(result, [50.0, 100.0])


if __name__ == '__main__':
    unittest.main()
