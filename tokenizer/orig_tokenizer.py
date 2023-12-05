import tensorflow as tf
import sentencepiece as spm

TRAINING_TEXT = 'intake/berte_per_sentence.txt'
MODEL = 'export/berte_tokenizer'
VOCAB_SIZE = 16000
BUFFER_SIZE = 20000
BATCH_SIZE = 64

spm.SentencePieceTrainer.train('''--input={}\
    --model_prefix={}\
    --vocab_size={}\
    --pad_id=0\
    --unk_id=1\
    --bos_id=2\
    --eos_id=3\
    --control_symbols=<mask>,<sep>,<cls>,<cls1>,<cls2>,<cls3>,<cls4>,<placeholder1>,<placeholder2>,<placeholder3>'''.format(
    TRAINING_TEXT, MODEL, VOCAB_SIZE))
