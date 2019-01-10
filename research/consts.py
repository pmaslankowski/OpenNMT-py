
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


MODEL_PATH = '/home/pma/Dropbox/Documents/Studia/Semestr_7/Praca_Inzynierska/OpenNMT-py/data/transformer/averaged-10-epoch.pt'
SENTENCEPIECE_MODEL_PATH = '/home/pma/Dropbox/Documents/Studia/Semestr_7/Praca_Inzynierska/OpenNMT-py/data/transformer/sentencepiece.model'
SOURCE_PATH = 'example.atok'
TARGET_PATH = 'example_target.atok'
OUTPUT_PATH = 'example_output.atok'
OPT = AttrDict({
            'models': [MODEL_PATH], 'data_type': 'text', 'src': SOURCE_PATH, 'src_dir': '',
            'tgt': TARGET_PATH, 'output': OUTPUT_PATH, 'report_bleu': False, 'report_rouge': False,
            'dynamic_dict': False, 'share_vocab': False, 'fast': False, 'beam_size': 5, 'min_length': 0, 'max_length': 100,
            'max_sent_length': None, 'stepwise_penalty': False, 'length_penalty': 'none', 'coverage_penalty': 'none',
            'alpha': 0.0, 'beta': -0.0, 'block_ngram_repeat': 0, 'ignore_when_blocking': [], 'replace_unk': False,
            'verbose': True, 'log_file': '', 'attn_debug': False, 'dump_beam': '', 'n_best': 1, 'batch_size': 100, 'gpu': -1,
            'sample_rate': 16000, 'window_size': 0.02, 'window_stride': 0.01, 'window': 'hamming', 'image_channel_size': 3,
            'temperature': 1.0, 'generator_function': 'temperature'
        })