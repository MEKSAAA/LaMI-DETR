from detrex.config import get_config
from .model.roi_convnextl import model

# get default config
dataloader = get_config("common/data/lvis_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train


# modify training config
train.init_checkpoint = "clip_convnext_large_trans.pth"
train.output_dir = "./output/roi_clip/idow_convnext_large_lvis"

# max training iterations
train.max_iter = 36000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# model.query_path = "dataset/metadata/lvis_a+name_convnextl.npy" 
# model.eval_query_path = "dataset/metadata/lvis_a+name_convnextl.npy"
# model.query_path = "dataset/metadata/lvis_visual_desc_convnextl.npy" 
# model.eval_query_path = "dataset/metadata/lvis_visual_desc_convnextl.npy"
# model.query_path = "dataset/metadata/lvis_visual_desc_confuse_cluster_convnextl.npy" 
# model.eval_query_path = "dataset/metadata/lvis_visual_desc_confuse_cluster_convnextl.npy"
# model.query_path = "dataset/metadata/lvis_visual_desc_confuse_convnextl.npy" 
# model.eval_query_path = "dataset/metadata/lvis_visual_desc_confuse_convnextl.npy"
model.query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy" 
model.eval_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy" 
# model.query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_top3_convnextl.npy" 
# model.eval_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_top3_convnextl.npy" 

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.mapper.is_train = True
dataloader.test.dataset.filter_empty=True