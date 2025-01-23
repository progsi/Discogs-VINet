import random
import torch
import torchaudio

class SpecAug(torch.nn.Module):
    def __init__(self, 

                 T: int = 40,
                 F: int = 30,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2,
                 noise_std: float = 0.01,
                 p_noise: float = 0.5,
                 replace_with_zero: bool = False):
        """
        Initialize the augmentation transform.
        Transforms are adopted from: https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb

        Args:
            T (int): Maximum width of the time mask.
            F (int): Maximum width of the frequency mask.
            num_time_masks (int): Number of time masks to apply.
            num_freq_masks (int): Number of frequency masks to apply.
            noise_std (float): Standard deviation of the Gaussian noise to add.
            p_noise (float): Probability of adding noise.
            replace_with_zero (bool): Whether to replace masked values with zero.
        """
        super().__init__()
        self.T = T
        self.F = F
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
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
        if torch.rand(1).item() < self.p_noise:
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
        spectrogram = self.apply_time_mask(spectrogram)
        spectrogram = self.apply_freq_mask(spectrogram)
        spectrogram = self.add_noise(spectrogram)
        return spectrogram
