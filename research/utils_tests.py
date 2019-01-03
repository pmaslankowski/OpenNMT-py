import unittest
from research.utils import *

class TestAligner(unittest.TestCase):

    def test_allignment(self):
        sen1 = ['▁Ich', '▁halte', '▁die', '▁', 'maschine', 'lle',
                '▁Übersetzung', '▁für', ' ▁ein', '▁sehr',
                '▁interessante', 's', '▁Thema', '.']
        sen2 = ['▁Ich', '▁denke', '▁dass', '▁die', '▁', 'maschine', 'lle',
                '▁Übersetzung', '▁sehr', '▁interessant', '▁ist', '.']

        expected1 = ['▁Ich', '▁halte', '[PLACEHOLDER]', '▁die', '▁', 'maschine', 'lle',
                     '▁Übersetzung', '▁für', ' ▁ein', '▁sehr', '▁interessante', 's', '▁Thema', '.']

        expected2 = ['▁Ich', '▁denke', '▁dass', '▁die', '▁', 'maschine', 'lle', '▁Übersetzung',
                     '[PLACEHOLDER]', '[PLACEHOLDER]', '▁sehr', '[PLACEHOLDER]', '▁interessant', '▁ist', '.']

        aligner = Aligner()
        res1, res2 = aligner.align(sen1, sen2)

        self.assertEquals(expected1, res1)
        self.assertEqual(expected2, res2)


class TestBleu(unittest.TestCase):

    def test_bleu(self):
        translation = '▁Ein - jährige - O ld - ▁Ge ▁Dead'
        reference = '▁28 - jährige r ▁Koch ▁in ▁San ▁Francisco ▁Mal l ▁to t ▁auf gefunden'
        expected = 0.0

        self.assertEquals(expected, bleu(reference, translation))

if __name__ == '__main__':
    unittest.main()
