import os
import shutil
import json

from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_text as text

def _copy_unowned(dst_dir, src_dir, owned_files):
    files = os.listdir(src_dir)
    for file in set(files).difference(set(owned_files)):
        src = os.path.join(src_dir, file)
        dst = os.path.join(dst_dir, file)
        if os.path.exists(src) and src != dst:
            # copy source to dest
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst)

class SaveableModule(tf.Module):
    """
    _saveableModule save and load elements of this module in a structured directory.
    """
    def __init__(self, src_dir, rws):
        super().__init__()

        self.src_dir = src_dir
        self.rws = rws

        self.elems = {
            elem_key: rws[elem_key].load(src_dir)
            for elem_key in rws
        }

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def save(self, dst_dir):
        """
        save models under model_path
        """

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        owned_files = []
        for elem_key in self.rws:
            writer = self.rws[elem_key]
            writer.save(self.elems[elem_key], dst_dir)
            owned_files += writer.owned_files()

        if os.path.exists(self.src_dir):
            _copy_unowned(dst_dir, self.src_dir, owned_files)

class SubmoduleReadWriter(ABC):
    """
    SubmoduleReadWriter saves and loads some element from files/directories and marks those
    files/directories as owned.
    """
    @abstractmethod
    def save(self, elem, dst_parent):
        """
        save the elem to some files/directories under dst_parent
        """

    @abstractmethod
    def load(self, src_parent):
        """
        load the element from owned files/directories under src_parent
        """

    @abstractmethod
    def owned_files(self):
        """
        return list of owned files/directories
        """

class KerasReadWriter(SubmoduleReadWriter):
    """
    KerasReadWriter saves and loads from keras models
    """
    def __init__(self, directory, kmodel, *args, **kwargs):

        self.directory = directory
        self.kmodel = kmodel

        self.backup_args = list(args)
        self.backup_kwargs = dict(kwargs)

    def save(self, elem, dst_parent):
        """
        save the keras model elem to join(dst_parent, directory)
        """

        dst_path = os.path.join(dst_parent, self.directory)
        elem.save(dst_path)

    def load(self, src_parent):
        """
        load the keras model elem from join(src_parent, directory)
        """

        src_path = os.path.join(src_parent, self.directory)
        if os.path.exists(src_path):
            return tf.keras.models.load_model(src_path)
        return self.kmodel(*self.backup_args, **self.backup_kwargs)

    def owned_files(self):
        """
        return owned directory name
        """

        return [self.directory]

class SModuleReadWriter(SubmoduleReadWriter):
    """
    SModuleReadWriter saves and loads from other SaveableModule
    """
    def __init__(self, directory, smodel, *args, **kwargs):

        self.directory = directory
        self.smodel = smodel

        self.args = list(args)
        self.kwargs = dict(kwargs)

    def save(self, elem, dst_parent):
        """
        save the keras model elem to join(dst_parent, directory)
        """

        dst_path = os.path.join(dst_parent, self.directory)
        elem.save(dst_path)

    def load(self, src_parent):
        """
        load the keras model elem from join(src_parent, directory)
        """

        src_path = os.path.join(src_parent, self.directory)
        return self.smodel(src_path, *self.args, **self.kwargs)

    def owned_files(self):
        """
        return owned directory name
        """

        return [self.directory]

class SaveableTokenizer:
    """
    SaveableTokenizer is the same as text.SentencepieceTokenizer but allows saving.
    """

    def __init__(self, src, tokenizer_setup):

        self.src = src
        if os.path.exists(src+'.model'):
            with open(src+'.model', 'rb') as file:
                self.tokenizer = text.SentencepieceTokenizer(
                        model=file.read(),
                        out_type=tf.int32,
                        add_bos=tokenizer_setup["add_bos"],
                        add_eos=tokenizer_setup["add_eos"])

            for func in dir(text.SentencepieceTokenizer):
                if callable(getattr(text.SentencepieceTokenizer, func)) and func[:2] != '__':
                    setattr(self.__class__, func, getattr(self.tokenizer, func))
        else:
            self.tokenizer = None

    def save(self, dst_parent):
        """
        save the tokenizer by moving around vocab and model files
        """

        dst = os.path.join(dst_parent, os.path.basename(self.src))
        shutil.copyfile(self.src+'.vocab', dst+'.vocab')
        shutil.copyfile(self.src+'.model', dst+'.model')

class TokenizerReadWriter(SubmoduleReadWriter):
    """
    TokenizerReadWriter saves and loads from text.SentencepieceTokenizer
    """

    def __init__(self, name, tokenizer_setup):

        self.name = name
        self.tokenizer_setup = tokenizer_setup

    def save(self, elem, dst_parent):
        """
        save the keras model elem to join(dst_parent, directory)
        """

        elem.save(dst_parent)

    def load(self, src_parent):
        """
        load the keras model elem from join(src_parent, directory)
        """

        src_path = os.path.join(src_parent, self.name)
        return SaveableTokenizer(src_path, self.tokenizer_setup)

    def owned_files(self):
        """
        return owned directory name
        """

        return [self.name + ext for ext in ['.vocab', '.model']]

class FileReadWriter(SubmoduleReadWriter):
    def __init__(self, filename, saver=None):

        self.filename = filename
        self.saver = saver

    def save(self, elem, dst_parent):
        """
        save the keras model elem to join(dst_parent, directory)
        """

        if self.saver is not None:
            self.saver(elem)

        dst_path = os.path.join(dst_parent, self.filename)
        with open(dst_path, 'w') as file:
            json.dump(elem, file)

    def load(self, src_parent):
        """
        load the keras model elem from join(src_parent, directory)
        """

        src_path = os.path.join(src_parent, self.filename)
        if os.path.exists(src_path):
            with open(src_path, 'r') as file:
                return json.loads(file.read())
        return dict()

    def owned_files(self):
        """
        return owned directory name
        """

        return [self.filename]
