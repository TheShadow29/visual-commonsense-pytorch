# visual-commonsense-pytorch
For visual commonsense model. Paper: https://arxiv.org/pdf/1811.10830.pdf

Note that this is unofficial implementation.

### Steps for getting Bert-results:
1. I am using pytorch pretrained bert from huggingface here: https://github.com/huggingface/pytorch-pretrained-BERT. So that needs to be installed first. 
1. Download data from http://visualcommonsense.com/download.html. You should have two folders - images and annotations.
1. `cd code`
1. Change the `cfg.json` file to set the `vcr_tdir`
1. `python bert_main.py "some_unique_id" --task_type $T`. The task type can be `QA` or `QA_R`. `Q_AR` is under progress.

### Results:
QA gives 59% on validation set, QA_R gives 66%, both of which are higher than what is reported in the paper (53% and 64%). I am not sure why this the case though. Any inputs are welcome. 
