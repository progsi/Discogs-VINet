import random
import torch
import torchaudio

class SpecAug(torch.nn.Module):
    def __init__(self, 
                 T: int = None,
                 num_time_masks: int = None,
                 p_time_mask: float = 1.0,
                 F: int = None,
                 num_freq_masks: int = None,
                 p_freq_mask: float = 1.0,
                 noise_std: float = None,
                 p_noise: float = None,
                 replace_with_zero: bool = False):
        """
        Initialize the augmentation transform.
        Transforms are adopted from: https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb

        Args:
            T (int, optional): Maximum width of the time mask. If None, no time mask will be applied.
            num_time_masks (int, optional): Number of time masks to apply. If None, no time mask will be applied.
            p_time_mask (float, optional): Probability of applying a time mask. Default is 1.0.
            F (int, optional): Maximum width of the frequency mask. If None, no frequency mask will be applied.
            num_freq_masks (int, optional): Number of frequency masks to apply. If None, no frequency mask will be applied.
            p_freq_mask (float, optional): Probability of applying a frequency mask. Default is 1.0.
            noise_std (float, optional): Standard deviation of the Gaussian noise to add. If None, no noise will be added.
            p_noise (float, optional): Probability of adding noise. If None, no noise will be added.
            replace_with_zero (bool): Whether to replace masked values with zero.
        """
        super().__init__()
        self.T = T
        self.num_time_masks = num_time_masks
        self.p_time_mask = p_time_mask
        self.F = F
        self.num_freq_masks = num_freq_masks
        self.p_freq_mask = p_freq_mask
        self.noise_std = noise_std
        self.p_noise = p_noise
        self.replace_with_zero = replace_with_zero

    def apply_time_mask(self, x):
        cloned = x.clone()
        len_spectro = cloned.shape[1]
        
        for i in range(0, self.num_time_masks):
            t = random.randrange(0, self.T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (self.replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned


    def apply_freq_mask(self, x):
        cloned = x.clone()
        num_channels = cloned.shape[0]
        
        for i in range(0, self.num_freq_masks):        
            f = random.randrange(0, self.F)
            f_zero = random.randrange(0, num_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (self.replace_with_zero): cloned[0][f_zero:mask_end] = 0
            else: cloned[f_zero:mask_end] = cloned.mean()
        
        return cloned

    def add_noise(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the spectrogram."""
        noise = torch.randn_like(spectrogram) * self.noise_std
        spectrogram += noise
        return spectrogram

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation to a given spectrogram.

        Args:
            spectrogram (torch.Tensor): Input spectrogram tensor of shape (channels, freq, time).

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        if self.T and self.num_time_masks and random.random() < self.p_time_mask:
            spectrogram = self.apply_time_mask(spectrogram)
        if self.F and self.num_freq_masks and random.random() < self.p_freq_mask:
            spectrogram = self.apply_freq_mask(spectrogram)
        if self.noise_std and self.p_noise and random.random() < self.p_noise:
            spectrogram = self.add_noise(spectrogram)
        return spectrogram
