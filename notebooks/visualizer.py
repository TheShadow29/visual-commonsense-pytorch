import PIL
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import json
from colormap import colormap


@dataclass
class Visualizer(object):
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
        self.color_list = colormap(rgb=True) / 255

    def get_img(self) -> Image:
        return PIL.Image.open(self.vcr_imgs / self.img_fn)

    def get_metadata(self) -> Dict:
        return json.load((self.vcr_imgs / self.meta_fn).open('r'))

    def get_mask_ann_img(self) -> Image:
        img = self.get_img()
        drawer = ImageDraw.Draw(img, 'RGBA')
        segms = self.meta_data['segms']
        for ind, seg in enumerate(segms):
            poly = seg[0]
            drawer.polygon(poly, fill=colormap[ind])
