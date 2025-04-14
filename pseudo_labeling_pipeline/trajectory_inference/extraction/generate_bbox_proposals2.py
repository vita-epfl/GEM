import os
import sys
from glob import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import curve_fit
import cv2
import random
import secrets
import shutil
from scipy import interpolate
from dataclasses import dataclass
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from transformers import  SegformerImageProcessor, SegformerForSemanticSegmentation, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
from geocalib import GeoCalib
from depth_anything_v2.metric_depth.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)

def ade_palette():
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [244, 35, 232],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [128, 64, 128],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [128, 64, 128],
      [120, 120, 70],
      [244, 35, 232],
      [255, 6, 82],
      [70, 70, 70],
      [204, 255, 4],
      [220, 20, 60],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [128, 64, 128],
      [128, 64, 128],
      [255, 7, 71],
      [255, 9, 224],
      [70, 130, 180],
      [220, 220, 220],
      [152, 251, 152],
      [107, 142, 35],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [128, 64, 128],
      [0, 255, 20],
      [128, 64, 128],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [250, 170, 30],
      [140, 140, 140],
      [220, 220, 0],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [0, 0, 142],
      [0, 0, 142],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [0, 0, 142],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
    #transforms.PILToTensor()
    transforms.ToTensor(),
])
transform_PIL = Compose([
    transforms.ToPILImage(),
])

#load segmentation model
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model_seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model_yolo = YOLO("yolo11n-seg.pt")
model_seg.to(device) 
model_yolo.to(device) 

image_processor_m2f = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
model_seg_m2f = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
model_seg_m2f.to(device)

#load depth model
image_processor_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
#image_processor_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
#model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
model_depth.to(device)
# load calib model
calib_model = GeoCalib()
calib_model.to(device)

# load depth model
# dataset = "vkitti"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
# max_depth = 80  # 20 for indoor model, 80 for outdoor model
# model_configs = {
#     "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
#     "vitb": {
#         "encoder": "vitb",
#         "features": 128,
#         "out_channels": [96, 192, 384, 768],
#     },
#     "vitl": {
#         "encoder": "vitl",
#         "features": 256,
#         "out_channels": [256, 512, 1024, 1024],
#     },
# }
# depth_model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
# depth_model.load_state_dict(
#     torch.load(
#         os.path.join(
#             weights_dir,
#             f"depth_anything_v2_metric_{dataset}_{encoder}.pth",
#         ),
#         map_location="cpu",
#     )
# )
# depth_model.to(device).eval()
# depth_transform = Compose(
#     [
#         Resize(
#             width=518,
#             height=518,
#             resize_target=False,
#             keep_aspect_ratio=True,
#             ensure_multiple_of=14,
#             resize_method="lower_bound",
#             image_interpolation_method=cv2.INTER_CUBIC,
#         ),
#         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         PrepareForNet(),
#     ]
# )

def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c

@dataclass
class CarPlacement:
    x: int  # int center x coordinate
    y: int  # int center y coordinate
    width: int  # int width of the bounding box
    height: int  # int height of the bounding box
    confidence: float  # float confidence score for this placement

class CarPlacementFinder:
    def __init__(
        self,
        margin_percent = 0.2,
        min_car_width = 40,
        max_car_width = 200,
        min_car_height = 30,
        max_car_height = 150,
        min_road_width_factor = 1.5  # minimum road width relative to car width
    ):
        self.margin_percent = margin_percent
        self.min_car_width = min_car_width
        self.max_car_width = max_car_width
        self.min_car_height = min_car_height
        self.max_car_height = max_car_height
        self.min_road_width_factor = min_road_width_factor

    def calculate_size_at_position(self, y, image_height):
        """Calculate car size based on y position (perspective effect)"""
        # Linear interpolation for size based on y position
        position_factor = 1 - (y / image_height)  # 1 at top, 0 at bottom
        
        # Calculate width and height with perspective effect
        width = int(self.min_car_width + (self.max_car_width - self.min_car_width) * (1 - position_factor))
        height = int(self.min_car_height + (self.max_car_height - self.min_car_height) * (1 - position_factor))
        
        return width, height

    def find_road_width_at_y(self, binary_mask, y):
        """Find road width at a specific y coordinate"""
        row = binary_mask[y, :]
        road_pixels = np.where(row > 0)[0]
        if len(road_pixels) == 0:
            return 0
        return road_pixels[-1] - road_pixels[0]

    def adjust_bbox(self, x_center, y_center, width, height, img_width, img_height):
        """
        Adjusts the bounding box width based on its position relative to the image center and y-coordinate.
        The height is kept the same.
        """
        x_dist = abs(x_center - img_width // 2)
        # Calculate the normalized y-coordinate (0 to 1, where 1 is the bottom of the image)
        norm_y = 1 - (y_center / img_height)
        # Adjust the width based on the distance from the image center and the normalized y-coordinate
        width_factor = 1 + 0.2 * (x_dist / (img_width // 2)) * (1 - norm_y)
        # Apply the width adjustment, but keep the height the same
        adjusted_width = width * width_factor
        adjusted_height = height
        return adjusted_width, adjusted_height

    def get_valid_placements(self, binary_mask, binary_car_mask, bbox_size_candidate, num_positions_y=40, num_positions_x=40):
        """Find valid car placements in the binary road mask"""
        height, width = binary_mask.shape
        margin_x = int(width * self.margin_percent)
        margin_y = int(height * self.margin_percent)

        #boundaries of the road
        y, x = np.where(binary_mask)
        # Find the minimum and maximum x and y coordinates
        road_x_min, road_x_max, road_y_min, road_y_max = x.min(), x.max(), y.min(), y.max()
    
        valid_placements = []
        
        # Create a distance transform of the road mask
        # This will help us keep cars centered on the road
        distance_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Sample y positions from bottom to top
        y_positions = np.linspace(height - margin_y, margin_y, num_positions_y).astype(int)
        
        for y in y_positions:
            
            if y not in bbox_size_candidate.keys():
                continue
            # Get car size at this y position
            car_width = np.array(bbox_size_candidate[y]['width']).mean()
            car_height = np.array(bbox_size_candidate[y]['height']).mean()
            if np.isnan(y) or np.isnan(car_width) or np.isnan(car_height):
                continue
            #car_width, car_height = self.calculate_size_at_position(y, height)
            road_width = self.find_road_width_at_y(binary_mask, y)
            
            # Skip if road is too narrow for car
            if road_width < car_width * self.min_road_width_factor:
                continue    
            # Find road boundaries at this y
            row = binary_mask[y, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) < 2:
                continue
                
            road_left = road_pixels[0]
            road_right = road_pixels[-1]

            valid_x_min = max(margin_x + car_width//2, road_left + car_width//2)
            valid_x_max = min(width - margin_x - car_width//2, road_right - car_width//2)
            
            if valid_x_max <= valid_x_min:
                continue
            
            # Sample potential x positions
            x_positions = np.linspace(valid_x_min, valid_x_max, num_positions_x).astype(int)
            
            for x in x_positions:
                margin = 30
                confidence = distance_transform[y, x] / np.max(distance_transform)
                car_width, car_height = self.adjust_bbox(x, y, car_width, car_height, 1024, 576)
                #the bbox proposal shouldn't overlap with a car
                if binary_car_mask[int(y-car_height):int(y), int(x-car_width//2):int(x+car_width//2)].sum() > 0:
                    continue
                if binary_mask[int(y), int(x-car_width//2-margin):int(x+car_width//2+margin)].prod()== 0:
                    continue
                placement = CarPlacement(
                    x=int(x),
                    y=int(y-car_height//2),
                    width=int(car_width),
                    height=int(car_height),
                    confidence=confidence
                )
                valid_placements.append(placement)
        
        return valid_placements

    def visualize_placements(self, img, placements):
        """Visualize car placements on the mask"""
        # Convert binary mask to RGB
        if img.ndim == 2:
            vis_img = np.stack([img * 255] * 3, axis=-1).astype(np.uint8)
        else:
            vis_img = img.copy()
        # Draw each placement
        for placement in placements:
            x1 = int(placement.x - placement.width // 2)
            y1 = int(placement.y - placement.height // 2)
            x2 = int(placement.x + placement.width // 2)
            y2 = int(placement.y + placement.height // 2)
            
            # Color based on confidence (green with varying intensity)
            color = (0, int(255 * placement.confidence), 0)
            try:
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255,0,0), 2)
                # Draw center point
                cv2.circle(vis_img, (placement.x, placement.y), 2, (0, 0, 255), -1)
            except:
                print('CV2 failed')
            
        return vis_img

def create_pixel_dict(yolo_bbox, avg_depth, calib_matrix):
    # Define the y range and initialize the dictionary
    non_nan_indices = np.flatnonzero(~np.isnan(avg_depth))
    y_min, y_max = non_nan_indices[0], 575
    pixel_dict = {y: {'width': [], 'height': []} for y in range(y_min, y_max + 1)}
    correction_dict = {y: {'ratio': []} for y in range(y_min, y_max + 1)}
    # Set default values for the min and max keys
    pixel_dict[y_min]['width'].append(10)
    pixel_dict[y_min]['height'].append(10)
    pixel_dict[y_max]['width'].append(400)
    pixel_dict[y_max]['height'].append(400)

    # If bounding box information exists, populate the dictionary
    if yolo_bbox is not None:
        yolo_bbox = np.array(yolo_bbox)
        for box in yolo_bbox:
            x_center, y_center, width, height = box
            y_start = int(y_center - height / 2)
            y_end = int(y_center + height / 2)
            if y_end in pixel_dict.keys():
                pixel_dict[y_end]['width'].append(width)
                pixel_dict[y_end]['height'].append(height)  
                for y in range(y_min, y_max + 1):
                    if y!=y_end:
                        pixel_dict[y]['width'].append(width*(avg_depth[y]/avg_depth[y_end]))
                        pixel_dict[y]['height'].append(height*(avg_depth[y]/avg_depth[y_end]))

    return pixel_dict


def run_car_placement(img, binary_mask, binary_car_mask, yolo_bbox, avg_depth, calib_matrix, img_name='scene1'):

    binary_mask = binary_mask > 0  # Convert to binary
    # Create finder instance
    finder = CarPlacementFinder()
    #Create bbox width and height proposals for each height using depth+YOLO and only YOLO predictions
    bbox_size_candidate = create_pixel_dict(yolo_bbox, avg_depth, calib_matrix)
    # Find valid placements
    placements = finder.get_valid_placements(binary_mask, binary_car_mask, bbox_size_candidate)
    # Visualize results
    vis_img = finder.visualize_placements(binary_mask, placements)

    # Save or display result
    cv2.imwrite(img_name + '_car_placements.png', vis_img)

    bbox_candidates = []
    for placement in placements:
        bbox_candidates.append(placement)
    
    if len(bbox_candidates)==0:
        return None
    else:
        #using depth
        img_depth = img.copy()
        bbox_final = secrets.choice(bbox_candidates)
        final_bbox_img = finder.visualize_placements(img_depth, [bbox_final])
        # height_obj = avg_height[bbox_final.y]
        # bottom_obj = (int(bbox_final.x), int(bbox_final.y+bbox_final.height/2))
        # top_obj = (int(bbox_final.x), int(bbox_final.y+bbox_final.height/2-height_obj))
        #cv2.line(final_bbox_img, bottom_obj, top_obj, color = (255, 0, 255), thickness=4)
        final_bbox_img = cv2.cvtColor(final_bbox_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_name + '_final_bbox.png', final_bbox_img)
    return (bbox_final.x, bbox_final.y, bbox_final.width, bbox_final.height)


def colored_segmentation(pred_seg, img_name='scene1'):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color
    Image.fromarray(color_seg.astype(np.uint8)).save(img_name+ '_segmentation.png')

def get_calib(img):
    img = transform(img).unsqueeze(0).to(device)
    #img = (img.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
    #img = torch.from_numpy(img).to(self.device)
    with torch.no_grad():
        result = calib_model.calibrate(
            img,
            camera_model="pinhole",
            shared_intrinsics=False,
        )
        intrinsics_np = result["camera"].K.mean(dim=0).cpu().numpy()
        return intrinsics_np

def get_depth(img, img_name='scene1'):
    img = torch.Tensor(img).unsqueeze(0)
    img_height = img.shape[1]
    img_width = img.shape[2]
    inputs = image_processor_depth(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_depth(**inputs)
        predicted_depth = outputs.predicted_depth
        # interpolate to original size
        prediction = nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(img_height, img_width),
            mode="bilinear",
            align_corners=True,)
        prediction = prediction[0,0].cpu().numpy()
        depth = (prediction * 255 / prediction.max())
        depth = Image.fromarray(depth.astype("uint8")).save(img_name + '_depth.png')
        return prediction
    
def get_segmentation(img, img_name='scene1'):
    #img should be HxWx3
    img = torch.Tensor(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    inputs = image_processor_m2f(images=img, return_tensors="pt", do_resize=False).to(device)      
    with torch.no_grad():
        outputs = model_seg_m2f(**inputs)
        pred_semantic_map = image_processor_m2f.post_process_semantic_segmentation(outputs, target_sizes=[(img_height, img_width)])[0]
       
        road_mask = (pred_semantic_map == 8) | (pred_semantic_map == 13) | (pred_semantic_map == 23) | (pred_semantic_map == 24) | (pred_semantic_map == 41) | (pred_semantic_map == 43)
        road_mask = (road_mask*255.0).cpu().numpy()
        road_car = (pred_semantic_map == 55) | (pred_semantic_map == 61) | (pred_semantic_map == 54) 
        road_car = (road_car*255.0).cpu().numpy()
        colored_segmentation(pred_semantic_map.cpu().numpy(), img_name)
        Image.fromarray(road_mask.astype(np.uint8)).save(img_name +'_segmentation_road.png')
        Image.fromarray(road_car.astype(np.uint8)).save(img_name +'_segmentation_car.png')

    return road_mask, road_car, pred_semantic_map

def get_yolo_boxes(img_yolo, img_name='scene1'):
    #transform img to a tensor with value 0-1 and size Bx3xHxW
    img_yolo = transform(img_yolo)
    img_yolo = img_yolo.unsqueeze(0)
    results_yolo = model_yolo(img_yolo.to(device))
    yolo_boxes = None
    try:
        yolo_cls = results_yolo[0].boxes.cls
        orig_shape = results_yolo[0].orig_shape
        yolo_shape = results_yolo[0].masks[0].data[0].shape
        yolo_boxes = results_yolo[0].boxes.xywh
        scale_x = orig_shape[0] / yolo_shape[0]
        scale_y = orig_shape[1] / yolo_shape[1]
        img_yolo_pil = transform_PIL(img_yolo[0])
        draw = ImageDraw.Draw(img_yolo_pil)
        for b_ind, bbox in enumerate(yolo_boxes):
            #select the car objects
            if yolo_cls[b_ind] == 2:
                x, y, width, height = bbox
                x1 = x - width // 2
                y1 = y - height // 2
                x2 = x + width // 2
                y2 = y + height // 2
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        img_yolo_pil.save(img_name +'_car_bbox.png')
    except:
        print('YOLO bbox NOT found!')
    if yolo_boxes is not None:
       yolo_boxes = yolo_boxes.cpu().numpy()
    return yolo_boxes

def get_proposals(img, im_name='scene1'):
    calib_matrix = get_calib(img)
    print(calib_matrix)
    yolo_bboxes = get_yolo_boxes(img, img_name=im_name)
    seg_road, seg_car, seg_general = get_segmentation(img, img_name=im_name)
    depth_pred = get_depth(img, img_name=im_name)
    depth_pred = (depth_pred - depth_pred.min())/(depth_pred.max()-depth_pred.min())
    binary_road = seg_road == 255
    binary_car = seg_car == 255
    depth_level = depth_pred.copy()
    depth_level[~binary_road] = torch.nan #Do not take into account non-road pixels
    avg_depth_per_h = np.nanmean(depth_level, axis=1) #Size H
    bbox_selected = run_car_placement(img, binary_road, binary_car, yolo_bboxes, avg_depth_per_h, calib_matrix, img_name=im_name)
    return bbox_selected
    
def main():
    base_folder = "/mydata/swissai/isinsu/test_scenes/"

    if os.path.exists('./bbox_proposals/'):
        shutil.rmtree('./bbox_proposals/')
    # Recreate the folder
    os.makedirs('./bbox_proposals/')
    # get the image paths
    image_files = []
    if os.path.isdir(base_folder):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = sorted(glob(os.path.join(base_folder, '**/*.%s'%(ext)), recursive=True))
            if len(files)>0:
                image_files.extend(files)
    
    for img_path in image_files:
        print('Processing : ', img_path)
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext)-1]
        img = np.array(Image.open(img_path).convert('RGB'))
        #img = transform(Image.open(img_path).convert('RGB'))
        print('Image shape: ', img.shape)
        print(f"Image min { img.min()} max {img.max()}")
        save_name = './bbox_proposals/' + img_name
        bbox = get_proposals(img, im_name=save_name)
        print(bbox)
    
if __name__ == '__main__':
    main()