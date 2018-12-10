import PIL
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
from pathlib import Path
import json
from colormap import colormap
from collections import OrderedDict


@dataclass
class Visualizer(object):
    """
    A visualizer object that takes one
    annotation as input
    """
    vcr_tdir: str
    ann: Dict
    vcr_imgs: str = 'vcr1images'

    def __post_init__(self) -> None:
        self.vcr_tdir = Path(self.vcr_tdir)
        self.vcr_imgs = self.vcr_tdir / self.vcr_imgs
        self.img_fn = self.ann['img_fn']
        self.meta_fn = self.ann['metadata_fn']
        # self.img = self.get_img()
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
                poly = seg[0]
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
        return [x if isinstance(x, str) else self.obj_list[x[0]]
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
