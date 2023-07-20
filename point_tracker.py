import os
import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from einops import rearrange




class PointTracker():
    def __init__(self, reference_image, reference_points, checkpoint_path):
        """
        reference_image: tensor or numpy array (height, width, channels)
        reference_points: tensor or numpy array (num_points, 2)
        """
        if isinstance(reference_image, np.ndarray):
            reference_image = torch.from_numpy(reference_image)
            print("HI")

        if isinstance(reference_points, np.ndarray):
            reference_points = torch.from_numpy(reference_points)

        self.reference_image = rearrange(reference_image, 'h w c -> 1 1 c h w').float()

        reference_points = torch.cat([torch.zeros(reference_points.shape[0], 1), reference_points], dim=1)
        self.query = torch.tensor(reference_points).unsqueeze(0)
        self.checkpoint_path = checkpoint_path

        self.model = CoTrackerPredictor(checkpoint=self.checkpoint_path).to('cuda')

    def track(self, video):
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        video = rearrange(video, 't h w c -> 1 t c h w').float()
        video = torch.cat([self.reference_image, video], dim=1)


        pred_tracks, pred_visibility = self.model(
            video.to('cuda'),
            queries = self.query.to('cuda'),
        )

        return pred_tracks.squeeze(0).cpu()
        
