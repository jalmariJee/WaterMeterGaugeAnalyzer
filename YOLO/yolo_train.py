import torch
import torchvision.transforms as T 
import os

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

os.current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN = os.path.join(os.current_dir, "train/")
PATH_VAL = os.path.join(os.current_dir, "test/")


class CustomizedDataset(ClassificationDataset):
    """ A Customized dataset class for image classification with enhanced data augmentation. """
    def __init__(self, root:str, args, augment:bool=False, prefix: str=""):
        """ Initialize a customized  classification dataset with enhances data augmentation transforms"""
        super().__init__(root,args,augment,prefix)
        # Add your custom training transform here
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                #T.RandomHorizontalFlip(p=args.fliplr),
                #T.RandomVerticalFlip(p=args.flipud),
                #T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                #T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                #T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                #T.RandomErasing(p=args.erasing, inplace = True),

            ]
        )

        # Add your custom validation transform here
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )

        self.torch_transforms = train_transforms if augment else val_transforms
    
class CustomizedTrainer(ClassificationTrainer):
    """ A Customized trainer class for image classification with enhanced data augmentation. """
    def build_dataset(self, img_path:str, mode:str = "train"):
        """ Build a customized dataset for image classification with enhanced data augmentation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=(mode=="train"), prefix=self.args.split)
    
class CustomizedValidator(ClassificationValidator):
    """ A Customized validator class for image classification with enhanced data augmentation. """
    def build_dataset(self, img_path:str, mode:str = "val"):
        """ Build a customized dataset for image classification with enhanced data augmentation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=(mode=="train"), prefix=self.args.split)
    
model = YOLO("yolo26n-cls.pt") # load a pretrained model (recommended for training)
model.train(data= PATH_TRAIN, trainer=CustomizedTrainer, epochs=10, batch=64, imgsz=224)
model.val(data=PATH_VAL , validator=CustomizedValidator, batch=64, imgsz=224)