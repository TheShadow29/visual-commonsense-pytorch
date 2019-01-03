"""
Data loader
"""
import jsonlines as jsnl
import _init_paths
from PIL import Image, ImageDraw, ImageFont
import PIL
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
import pandas as pd
import ast
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pickle
import logging
from utils import DataWrap
fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
logger = logging.getLogger(__name__)


def pil2tensor(image: Image, dtype: np.dtype = np.float_)-> torch.tensor:
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
        a = np.transpose(a, (1, 0, 2))
        a = np.transpose(a, (2, 1, 0))
        return torch.from_numpy(a.astype(dtype, copy=False))


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choice_feats
        ]
        for feature in features
    ]


@dataclass
class InputFeatures:
    example_id: int
    choice_feats: List[Dict]
    label: int


@dataclass
class VCRDataset(Dataset):
    cfg: Dict[str, Any]
    csv_file: str

    def __post_init__(self):
        self.vcr_tdir = Path(self.cfg['vcr_tdir'])
        self.vcr_imgs = self.vcr_tdir / self.cfg['vcr_imgs']
        self.vcr_annots = self.vcr_tdir / self.cfg['vcr_annots']
        self.csv_file = Path(self.csv_file)
        self.task_type = self.cfg['task_type']
        self.sent_cache_file = self.vcr_annots / \
            f'{self.csv_file.stem}_sent_feats_{self.task_type}.pkl'
        # Prepare csv file if it doesn't exist
        if not self.csv_file.exists():
            logger.info('Converting jsonl files to pickle format')
            jsnl_file = self.csv_file.parent / \
                f"{self.csv_file.stem.split('_')[0]}.jsonl"
            jsnl_reader = jsnl.Reader(jsnl_file.open('r'))
            jsnl_data = [k for k in tqdm(jsnl_reader.iter())]
            jsnl_data_df = pd.DataFrame(jsnl_data)
            # jsnl_data_df.to_csv(self.csv_file, index=False)
            jsnl_data_df.to_pickle(self.csv_file)
        self.csv_data = pd.read_pickle(self.csv_file)

        # Prepare sentence file if not present
        if not self.sent_cache_file.exists():
            logger.info('Preprocessing the sentence features')
            self.tokenizer = BertTokenizer.from_pretrained(
                self.cfg['bert_model'], do_lower_case=self.cfg['do_lower_case'])
            if self.task_type == 'QA':
                self.max_seq_len = 200
                self.preprocess_sents_QA()
            elif self.task_type == 'QA_R':
                self.max_seq_len = 400
                self.preprocess_sents_QA_R()
            elif self.task_type == 'Q_AR':
                self.preprocess_sents_Q_AR()
        self.sent_data = pickle.load(self.sent_cache_file.open('rb'))
        logger.info('Loaded sentence data')

    def get_ann_from_idx(self, idx):
        return self.csv_data.iloc[idx]

    def create_obj_counter(self, ann: Dict[str, Any]) -> [Dict, List]:
        "create an object dictionary and list"
        objs = ann['objects']
        obj_dict = {}
        obj_list = []
        for obj in objs:
            if obj in obj_dict:
                obj_dict[obj] += 1
            else:
                obj_dict[obj] = 1

            tmp = obj + str(obj_dict[obj])
            obj_list.append(tmp)

        return obj_dict, obj_list

    def get_img(self, img_fn: str) -> Image:
        "Get the required PIL Image"
        return PIL.Image.open(self.vcr_imgs / img_fn)

    def get_metadata(self, meta_fn: str) -> Dict:
        "Get the required meta data"
        return json.load((self.vcr_imgs / meta_fn).open('r'))

    def get_question(self, ann: Dict[str, Any], obj_list: List[List]) -> str:
        "Get the question for the image"
        qu_text = ann['question']
        qu = self.replace_obj_names(qu_text, obj_list)
        return ' '.join(qu)

    def replace_obj_names(self, inp: List, obj_list: List) -> List[str]:
        "Converts stuff like [2] to the relevant object name"
        return [x if isinstance(x, str) else ' '.join([obj_list[y] for y in x])
                for x in inp]

    def get_answers(self, ann: Dict, obj_list: List) -> List[str]:
        "Returns the answer choices"
        ans_choices = ann['answer_choices']
        return [' '.join(self.replace_obj_names(a, obj_list)) for a in ans_choices]

    def get_reasons(self, ann: Dict, obj_list: List) -> List[str]:
        "Returns the rationale choices"
        reasons = ann['rationale_choices']
        return [' '.join(self.replace_obj_names(r, obj_list)) for r in reasons]

    def get_correct_answer_reason(self, ann: Dict) -> List[str]:
        "Returns the correct answer, rationale labels"
        return [ann['answer_label'], ann['rationale_label']]

    def get_QAR(self, ann: Dict, obj_list: List) -> Dict[str, Union[List[str], str]]:
        "Question, Answers, Reasoning, Labels at once"
        out_dict = OrderedDict()
        out_dict['question'] = self.get_question(ann, obj_list)
        out_dict['answer_choices'] = self.get_answers(ann, obj_list)
        out_dict['answer_label'] = ann['answer_label']
        out_dict['rationale_choices'] = self.get_reasons(ann, obj_list)
        out_dict['rationale_label'] = ann['rationale_label']
        return out_dict

    def preprocess_sents_QA(self):
        """
        Preprocess sentences. Currently only QA
        """
        features = []
        for idx in tqdm(range(len(self.csv_data))):
            ann = self.get_ann_from_idx(idx)
            obj_dict, obj_list = self.create_obj_counter(ann)
            qar = self.get_QAR(ann, obj_list)
            qu = qar['question']
            ans = qar['answer_choices']
            ans_label = qar['answer_label']
            qu_toks = self.tokenizer.tokenize(qu)
            choice_feats = []
            for ans_choice in ans:
                ans_choice_toks = self.tokenizer.tokenize(ans_choice)
                tokens = ["[CLS]"] + qu_toks + \
                    ["[SEP]"] + ans_choice_toks + ["[SEP]"]
                segment_ids = [0] * (len(qu_toks) + 2) +\
                    [1] * (len(ans_choice_toks) + 1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1]*len(input_ids)

                padding = [0] * (self.max_seq_len - len(input_ids))
                tot_len = len(input_mask)
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                try:
                    assert len(input_ids) == self.max_seq_len
                    assert len(input_mask) == self.max_seq_len
                    assert len(segment_ids) == self.max_seq_len
                except:
                    import pdb
                    pdb.set_trace()
                choice_feats.append({'input_ids': input_ids,
                                     'input_mask': input_mask,
                                     'segment_ids': segment_ids,
                                     'tot_len': tot_len})

            features.append(InputFeatures(example_id=idx,
                                          choice_feats=choice_feats, label=ans_label))

        with self.sent_cache_file.open('wb') as f:
            pickle.dump(features, f)

    def preprocess_sents_QA_R(self):
        """
        Preprocess sentences. For QA -> R
        """
        features = []
        for idx in tqdm(range(len(self.csv_data))):
            ann = self.get_ann_from_idx(idx)
            obj_dict, obj_list = self.create_obj_counter(ann)
            qar = self.get_QAR(ann, obj_list)
            qu = qar['question']
            ans = qar['answer_choices']
            ans_label = qar['answer_label']
            corr_ans = ans[ans_label]
            rat = qar['rationale_choices']
            rat_label = qar['rationale_label']
            final_qa = qu + corr_ans
            qu_toks = self.tokenizer.tokenize(final_qa)
            choice_feats = []
            for rat_choice in rat:
                rat_choice_toks = self.tokenizer.tokenize(rat_choice)
                tokens = ["[CLS]"] + qu_toks + \
                    ["[SEP]"] + rat_choice_toks + ["[SEP]"]
                segment_ids = [0] * (len(qu_toks) + 2) +\
                    [1] * (len(rat_choice_toks) + 1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1]*len(input_ids)

                padding = [0] * (self.max_seq_len - len(input_ids))
                tot_len = len(input_mask)
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                try:
                    assert len(input_ids) == self.max_seq_len
                    assert len(input_mask) == self.max_seq_len
                    assert len(segment_ids) == self.max_seq_len
                except:
                    import pdb
                    pdb.set_trace()
                choice_feats.append({'input_ids': input_ids,
                                     'input_mask': input_mask,
                                     'segment_ids': segment_ids,
                                     'tot_len': tot_len})

            features.append(InputFeatures(example_id=idx,
                                          choice_feats=choice_feats, label=rat_label))

        with self.sent_cache_file.open('wb') as f:
            pickle.dump(features, f)

    def get_BertDataset(self):
        "Returns a TensorDataset. Used for bert stuff"
        train_features = self.sent_data
        all_input_ids = torch.tensor(select_field(
            train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(
            train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(
            train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor(
            [f.label for f in train_features], dtype=torch.long)
        tot_len = torch.tensor(select_field(
            train_features, 'tot_len'), dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, tot_len)

    def __getitem__(self, idx):
        ann = self.get_ann_from_idx(idx)
        # img_fn = ann['img_fn']
        # img = pil2tensor(self.get_img(img_fn))
        # meta_fn = ann['metadata_fn']
        # meta_data = self.get_metadata(meta_fn)
        obj_dict, obj_list = self.create_obj_counter(ann)
        qar = self.get_QAR(ann, obj_list)
        out_dict = {}
        # out_dict['img'] = img
        # out_dict['meta_data'] = meta_data
        # out_dict['obj_list'] = obj_list
        # out_dict['ann'] = ann
        out_dict.update(qar)
        return out_dict


max_seq_len = 70


def bert_collater(batch):
    "Collater for simple bert"
    out_dict = {}
    out_lens = [b[4].max() for b in batch]
    max_inp_len = min(max(out_lens), max_seq_len)
    for ind in range(len(batch)):
        batch[ind] = list(batch[ind])
        for i in range(3):
            batch[ind][i] = batch[ind][i][:, :max_inp_len]

    out_dict['input_ids'] = torch.stack([b[0] for b in batch])
    out_dict['input_mask'] = torch.stack([b[1] for b in batch])
    out_dict['segment_ids'] = torch.stack([b[2] for b in batch])
    out_dict['target_labels'] = torch.stack([b[3] for b in batch])
    out_dict['tot_len'] = torch.stack([b[4] for b in batch])
    return out_dict


def get_vcr_dataset(cfg, set_type='train'):
    vcr_tdir = Path(cfg['vcr_tdir'])
    trn_file = vcr_tdir / cfg['vcr_annots'] / f'{set_type}_vcr.pkl'
    return VCRDataset(cfg, trn_file)


def get_bert_dataset(cfg, set_type='train'):
    vcr_train_ds = get_vcr_dataset(cfg, set_type)
    return vcr_train_ds.get_BertDataset()


def get_bert_data(cfg):
    bs = cfg['bs']
    nw = cfg['nw']

    train_ds = get_bert_dataset(cfg, 'train')
    train_dl = DataLoader(train_ds, batch_size=bs,
                          shuffle=True, num_workers=nw, drop_last=True, collate_fn=bert_collater)
    valid_ds = get_bert_dataset(cfg, 'val')
    valid_dl = DataLoader(valid_ds, batch_size=bs,
                          shuffle=False, num_workers=nw, drop_last=False, collate_fn=bert_collater)
    # test_ds = get_bert_dataset(cfg, 'test')
    # test_dl = DataLoader(test_ds, batch_size=bs,
    #                      shuffle=False, num_workers=nw, drop_last=False, collate_fn=bert_collater)
    path = Path('./tmp')
    data_bert = DataWrap(path=path, train_dl=train_dl,
                         valid_dl=valid_dl)
    return data_bert
    # test_ds = get_bert_dataset(cfg, 'test')


if __name__ == '__main__':
    cfg = json.load(open('./cfg.json'))
    # vcr_tdir = Path(cfg['vcr_tdir'])
    # trn_file = vcr_tdir / cfg['vcr_annots'] / 'train_vcr.pkl'
    # vcr_train_ds = VCRDataset(cfg, trn_file)
    # train_ds_bert = vcr_train_ds.get_BertDataset()
    logger.info('Preparing the Data')
    data = get_bert_data(cfg)
    batch = next(iter(data.train_dl))
