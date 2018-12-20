"""
Data loader
"""
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from colormap import colormap
from collections import OrderedDict
from dataclasses import dataclass
import PIL
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


@dataclass
class VCRDataset(Dataset):
    cfg: Dict[str, Any]
    csv_file: str

    def __post_init__(self):
        self.vcr_tdir = Path(self.cfg['vcr_tdir'])
        self.vcr_imgs = self.vcr_tdir / self.cfg['vcr_imgs']
        self.csv_file = Path(self.csv_file)
        self.csv_data = pd.read_csv(self.csv_file)

    def get_ann_from_idx(self, idx):
        return self.csv_data[idx]

    def __getitem__(self, idx):
        ann = self.get_ann_from_idx(idx)
        self.img_fn = ann['img_fn']
        self.meta_fn = ann['metadata_fn']
        self.meta_data = self.get_metadata()
        self.color_list = colormap(rgb=True).astype(np.int_)
        self.transparency = 128
        self.create_obj_counter()

    def create_obj_counter(self) -> None:
        "create an object dictionary and list"
        objs = self.ann['objects']
        obj_dict = {}
        obj_list = []
        for obj in objs:
            if obj in obj_dict:
                obj_dict[obj] += 1
            else:
                obj_dict[obj] = 1

            tmp = obj + str(obj_dict[obj])
            obj_list.append(tmp)

        self.obj_dict = obj_dict
        self.obj_list = obj_list
        return

    def get_img(self) -> Image:
        "Get the required PIL Image"
        return PIL.Image.open(self.vcr_imgs / self.img_fn)

    def get_metadata(self) -> Dict:
        "Get the required meta data"
        return json.load((self.vcr_imgs / self.meta_fn).open('r'))

    def get_mask_ann_img(self, draw_box: bool = True,  draw_mask: bool = True,
                         draw_text: bool = True) -> Image:
        "Annotate the image with ground truth masks"
        img = self.get_img()
        drawer = ImageDraw.Draw(img, 'RGBA')
        if draw_mask:
            segms = self.meta_data['segms']
            for ind, seg in enumerate(segms):
                # import pdb
                # pdb.set_trace()
                for poly in seg:
                    if len(poly) < 3:
                        continue
                    poly_tuple = [tuple(p) for p in poly]
                    color = self.color_list[ind].tolist()
                    drawer.polygon(
                        poly_tuple, fill=tuple(color + [self.transparency]))

        if draw_box:
            boxs = self.meta_data['boxes']
            fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
            font_to_use = ImageFont.truetype(fontPath, 32)
            for ind, box in enumerate(boxs):
                color = self.color_list[ind].tolist()
                drawer.rectangle(box[:-1], outline=tuple(color), width=3)
                if draw_text:
                    xy = box[:2]
                    text = self.obj_list[ind]
                    drawer.text(xy, text, font=font_to_use)
        return img

    def get_question(self) -> str:
        "Get the question for the image"
        qu_text = self.ann['question']
        qu = self.replace_obj_names(qu_text)
        return ' '.join(qu)

    def replace_obj_names(self, inp: List) -> List[str]:
        "Converts stuff like [2] to the relevant object name"
        return [x if isinstance(x, str) else ' '.join([self.obj_list[y] for y in x])
                for x in inp]

    def get_answers(self) -> List[str]:
        "Returns the answer choices"
        ans_choices = self.ann['answer_choices']
        return [' '.join(self.replace_obj_names(a)) for a in ans_choices]

    def get_reasons(self) -> List[str]:
        "Returns the rationale choices"
        reasons = self.ann['rationale_choices']
        return [' '.join(self.replace_obj_names(r)) for r in reasons]

    def get_correct_answer_reason(self) -> List[str]:
        "Returns the correct answer, rationale labels"
        return [self.ann['answer_label'], self.ann['rationale_label']]

    def get_QAR(self) -> Dict[str, Union[List[str], str]]:
        "Question, Answers, Reasoning, Labels at once"
        out_dict = OrderedDict()
        out_dict['question'] = self.get_question()
        out_dict['answer_choices'] = self.get_answers()
        out_dict['answer_label'] = self.ann['answer_label']
        out_dict['rationale_choices'] = self.get_reasons()
        out_dict['rationale_label'] = self.ann['rationale_label']
        return out_dict
