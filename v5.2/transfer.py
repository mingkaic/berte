import os
import logging
import shutil
import json

import yaml

import tensorflow as tf
import numpy as np

import export.model as model
import export.model3 as model3
import export.training as local_training

import common.nextsentence as ns
import common.training as training
import common.readwrite as rw

def convert_memory_processor(dirpath, params):
    oldprocessor = tf.keras.models.load_model(dirpath)
    newprocessor = model3.MemoryTextProcessor('', params)

    newembedder = model3.TextEmbed(params)
    newembedder.embedder = oldprocessor.embedder
    newembedder.encoder = oldprocessor.encoder

    newprocessor.elems['text_embedder'] = newembedder
    newprocessor.elems['memory'] = oldprocessor.mem
    newprocessor.elems['text_extruder'] = oldprocessor.perceiver

    newprocessor.embedder = newprocessor.elems['text_embedder']
    newprocessor.mem = newprocessor.elems['memory']
    newprocessor.decoder = newprocessor.elems['text_extruder']
    return newprocessor

def convert_textberte(src_path, optimizer, params, tokenizer_setup):
    src_metadata = os.path.join(src_path, 'metadata')
    src_processor = os.path.join(src_path, 'processor')

    out = model3.TextBert('', optimizer, params, tokenizer_setup)

    with open(src_metadata, 'r') as file:
        metadata = json.loads(file.read())
    tokenizer = rw.TokenizerReadWriter('tokenizer', tokenizer_setup).load(src_path)
    memory = convert_memory_processor(src_processor, params)
    predictor = rw.KerasReadWriter('text_predictor', model.Predictor,
            params['model_dim'], params['vocab_size']).load(src_path)

    out.elems['text_metadata'] = metadata
    out.elems['tokenizer'] = tokenizer
    out.elems['memory_processor'] = memory
    out.elems['text_predictor'] = predictor

    out.metadata = out.elems['text_metadata']
    out.tokenizer = out.elems['tokenizer']
    out.processor = out.elems['memory_processor']
    out.predictor = out.elems['text_predictor']

    return out

def mask_lm(bert, tokens, mask_rate,
        donothing_prob=0.1, rand_prob=0.1):
    """ given padded tokens, return mask_rate portion of tokens """

    mask = int(bert.metadata['mask'])
    lengths = np.array([len(sent) for sent in tokens])
    sizes = np.floor(mask_rate * lengths).astype(np.int32)

    out = []
    for (sent, size, length) in zip(tokens, sizes, lengths):
        if size < 1:
            out.append(sent)
            continue

        indices = np.random.choice(length, size=size, replace=False)
        state_probs = np.random.rand(size)

        values_info = [(
            index,
            (np.random.rand() if state_prob < donothing_prob + rand_prob else mask)
        ) for index, state_prob in zip(indices, state_probs) if state_prob < donothing_prob]
        if len(values_info) < 1:
            out.append(sent)
            continue

        indices = [index for index, _ in values_info]
        fill_values = [fill_value for _, fill_value in values_info]
        sent[indices] = fill_values
        out.append(sent)

    return out

def select_sentence_pairs(bert, token_windows, batch_size, window_size):
    """ given token windows and length windows, return first and second sentences selected within the token window """

    metadata = bert.metadata

    firsts = np.random.randint(low=0, high=window_size-1, size=(batch_size))
    seconds = np.array([ns.choose_second_sentence(window_size, first) for first in firsts])

    out_tokens = []
    s1_toks = []
    s2_toks = []
    length = 0
    for first_index, second_index, tokens in zip(firsts, seconds, token_windows):
        sentence1 = tokens[first_index]
        sentence2 = tokens[second_index]
        toks = np.concatenate(([int(metadata['cls'][0])], sentence1, [int(metadata['sep'])], sentence2), axis=0)
        out_tokens.append(toks)
        s1_toks.append(sentence1)
        s2_toks.append(sentence2)

    return out_tokens, firsts, seconds

def build_tester(bert, paragraph, logger, batch_size=15, mask_rate=0.15):
    """ build_tester returns a callable that tests the samples """
    window_size = len(paragraph)
    intokens = [bert.tokenize(sentence).numpy() for sentence in paragraph]
    masked = mask_lm(bert, intokens, mask_rate)
    masked_window = [masked] * batch_size
    tokens, s1_indices, s2_indices =\
        select_sentence_pairs(bert, masked_window, batch_size, len(paragraph))
    tokens = local_training._to_tensor(tokens, int(bert.metadata['pad']))
    distance_class = local_training._dist_class(s1_indices, s2_indices, window_size)

    def tester():
        prediction, _ = bert.predict(tokens, training=False)
        for s1, s2, tok, pred, dist_class in zip(
                s1_indices, s2_indices, tokens.numpy(), prediction.numpy(), distance_class):
            labels = np.argmax(pred, axis=-1).astype(np.int32)
            text = bert.tokenizer.detokenize(labels)

            logger.info('{:<25}: {}'.format("Input1", paragraph[s1]))
            logger.info('{:<25}: {}'.format("Input2", paragraph[s2]))
            logger.info('{:<25}: {}'.format("Tokens", tok))
            logger.info('{:<25}: {}'.format("Prediction Prob", pred))
            logger.info('{:<25}: {}'.format("Prediction Labels", labels))
            logger.info('{:<25}: {}'.format("Prediction Text", text))
            logger.info('{:<25}: {}'.format("Actual Dist Label", dist_class))
    return tester

if __name__ == '__main__':
    SRC = 'intake0/berte_pretrain'
    DST = 'intake/berte_pretrain'
    TMP_DIR = 'transfer_tmp'

    os.makedirs(TMP_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(TMP_DIR, 'converter.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    with open("configs/bert_pretrain.yaml") as file:
        _args = yaml.safe_load(file.read())
        tokenizer_setup = _args["tokenizer_args"]
        params = _args["model_args"]
    params['vocab_size'] = 1000

    if not os.path.exists(DST):
        os.makedirs(DST, exist_ok=True)

    learning_rate = training.CustomSchedule(params['model_dim'])
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    bert = convert_textberte(SRC, optimizer, params, tokenizer_setup)
    bert.metadata['add_bos'] = tokenizer_setup['add_bos']
    bert.metadata['add_eos'] = tokenizer_setup['add_eos']

    run_test = build_tester(bert, paragraph=[
        "the history of raja ampat is steeped in history and mythology.",
        "raja ampat was originally a part of an influential southeastern sultanate, the sultanate of tidore.",
        "in 1562, it was captured by the dutch, who influenced much of its cuisines and its architecture.",
        "that’s the more boring origin story.",
        "a colorful local tale talks of a woman who was given seven precious eggs, of which four (‘ampat’ in bahasa indonesian) hatched into kings (‘raja’).",
    ], logger=logger)
    run_test()

    bert.save(DST)
