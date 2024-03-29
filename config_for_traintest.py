#test指令
 python tools/test.py work_dirs/mam_faster-rcnn/20240131_091759/vis_data/config.py work_dirs/mam_faster-rcnn/result.pth
    
#train指令
python tools/train.py projects/MAM/configs/mam_faster-rcnn.py

# FeatmapAM method
python demo/vis/boxam_vis_demo.py pic.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
 
# EigenCAM method
python demo/vis/boxam_vis_demo.py pic.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method eigencam
 
# AblationCAM method
python demo/vis/boxam_vis_demo.py pic.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method ablationcam
 
# AblationCAM method and save img
python demo/vis/boxam_vis_demo.py pic.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method ablationcam --out-dir save_dir
 
# GradCAM
python demo/vis/boxam_vis_demo.py pic.jpg /.../config.py /.../result.pth --method gradcam
