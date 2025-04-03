import logging
from utils.custom_sampling import FrameSampler

class SamplingDatasetMixin:
    """Mixin class to add sampling index tracking to existing dataset classes."""
    
    def init_sampler(self, csv_save_dir, logger, dataset_name, split, sampling_method=None):
        """Initialize the FrameSampler for tracking sampling indices."""
        self.frame_sampler = FrameSampler(csv_save_dir, logger)
        self.dataset_name = dataset_name
        self.split = split
        # Get the appropriate sampling method based on the split
        if sampling_method:
            self.sampling_method = sampling_method
        elif hasattr(self, 'cfg') and hasattr(self.cfg, 'DATA'):
            # Use the config if available
            if split == "train" and hasattr(self.cfg.DATA, 'TRAIN_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.TRAIN_SAMPLING_METHOD
            elif split == "val" and hasattr(self.cfg.DATA, 'VAL_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.VAL_SAMPLING_METHOD
            elif hasattr(self.cfg.DATA, 'TEST_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.TEST_SAMPLING_METHOD
            else:
                self.sampling_method = 'uniform'  # Default
        else:
            self.sampling_method = 'uniform'  # Default
    
    def get_sampled_frames(self, video_path, total_frames, num_frames, sampling_method):
        """Get sampled frame indices and track them for CSV export."""
        return self.frame_sampler.sample_frames(
            video_path, 
            total_frames, 
            num_frames, 
            sampling_method,
            dataset_name=self.dataset_name,
            seed=42  # Fixed seed for reproducibility
        )
    
    def save_sampling_indices(self):
        """Save the sampling indices collected so far."""
        if hasattr(self, 'frame_sampler'):
            self.frame_sampler.save_indices_by_dataset(self.dataset_name, self.split)