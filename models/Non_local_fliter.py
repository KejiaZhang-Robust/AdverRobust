import torch
import torch.nn as nn

class GaussianNonLocalMeans(nn.Module):
    def __init__(self, in_channels):
        super(GaussianNonLocalMeans, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        theta = self.theta(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        phi = self.phi(x).view(batch_size, -1, height * width)
        f = torch.matmul(theta, phi)
        f_div_C = f.softmax(dim=-1)
        y = torch.matmul(f_div_C, x.view(batch_size, -1, height * width).permute(0, 2, 1))
        y = y.permute(0, 2, 1).view(batch_size, channels, height, width)
        y = self.conv1x1(y)
        return y


class DotProductNonLocalMeans(nn.Module):
    def __init__(self):
        super(DotProductNonLocalMeans, self).__init__()
        self.conv1x1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        f = torch.matmul(x.view(batch_size, channels, -1).permute(0, 2, 1), x.view(batch_size, channels, -1))
        f_div_C = f / (height * width)
        
        y = torch.matmul(x.view(batch_size, channels, -1), f_div_C).view(batch_size, channels, height, width)

        y = self.conv1x1(y)

        return y

class FFT_1D_NonLocal_Means(nn.Module):
    def __init__(self, K) -> None:
        super(FFT_1D_NonLocal_Means, self).__init__()
        self.K = K

    def forward(self, x):
        feature_maps = x
        batch_size, num_features, height, width = feature_maps.shape
        
        feature_maps = feature_maps.view(batch_size, num_features, -1)
        feature_maps_fft = torch.fft.fft(feature_maps, dim=2)
        freqs = torch.fft.fftfreq(feature_maps_fft.shape[-1])

        _, idx = torch.topk(freqs.abs(), self.K, largest=False)  

        mask = torch.zeros_like(feature_maps_fft, dtype=torch.bool)
        mask[:, :, idx] = 1

        feature_maps_fft_fil = feature_maps_fft.where(mask, torch.zeros_like(feature_maps_fft))
        feature_maps_recon = torch.fft.ifft(feature_maps_fft_fil, dim=2).real 
        
        return feature_maps_recon.view(batch_size, num_features, height, width)


class MeanFilter(nn.Module):
    def __init__(self, kernel_size=1):
        super(MeanFilter, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # Implement mean filter using 2D average pooling
        return nn.AvgPool2d(self.kernel_size, 1)(x) 

class MedianFilter(nn.Module):
    def __init__(self, kernel_size=1):
        super(MedianFilter, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # Unfold the feature map according to kernel size
        x_unfolded = torch.nn.functional.unfold(x, self.kernel_size, 1, (self.kernel_size-1)//2, 0)
        
        # Calculate median, dim=1 calculates median in neighborhoods
        # Median is calculated separately for each channel
        x_median = torch.median(x_unfolded, dim=1)[0]
        
        # Reshape result back to original shape
        x_median = x_median.view_as(x)
        
        return x_median
    


def shuffle_high_freqs(feature_maps):
    
    N, C, H, W = feature_maps.shape
    
    fmaps_fft = torch.fft.fftn(feature_maps, dim=(2,3))
    
    high_freqs = torch.fft.fftshift(fmaps_fft, dim=(2,3))
    random_idx = torch.randperm(C)
    high_freqs = high_freqs[:, random_idx]
    
    fmaps_fft[:, random_idx, H//2-1:H//2+1, W//2-1:W//2+1] = high_freqs[:, :, H//2-1:H//2+1, W//2-1:W//2+1]
    fmaps_shuffled = torch.fft.ifftn(fmaps_fft, dim=(2,3)).real

    return fmaps_shuffled

def feature_diff(X, HF):

    X_fft = torch.fft.fft2(X, dim=(-2, -1))
    HF_fft = torch.fft.fft2(HF, dim=(-2, -1))

    diff_fft = X_fft - HF_fft

    diff = torch.fft.ifft2(diff_fft, dim=(-2, -1)).real

    return diff

def reconstruct_feature_pytorch(feature_maps, k):

    batch_size, num_features, height, width = feature_maps.shape
    
    feature_maps = feature_maps.view(batch_size, num_features, -1)
    feature_maps_fft = torch.fft.fft(feature_maps, dim=2)
    freqs = torch.fft.fftfreq(feature_maps_fft.shape[-1])

    _, idx = torch.topk(freqs.abs(), k, largest=False)  

    mask = torch.zeros_like(feature_maps_fft, dtype=torch.bool)
    mask[:, :, idx] = 1

    feature_maps_fft_fil = feature_maps_fft.where(mask, torch.zeros_like(feature_maps_fft))
    feature_maps_recon = torch.fft.ifft(feature_maps_fft_fil, dim=2).real 
    
    return feature_maps_recon.view(batch_size, num_features, height, width)

def low_pass_2D_FFT(feature_maps, k):

  # Get shape of feature maps
  batch_size, num_features, height, width = feature_maps.shape  

  # Take 2D FFT over height and width dimensions
  fft2 = torch.fft.fft2(feature_maps, dim=(2, 3))

  # Generate 2D matrix of frequencies
  freqs = torch.fft.fftfreq2(height, width)
  _, idx = torch.topk(freqs, k, largest=False)

  # Create 2D mask to filter out frequencies
  mask = torch.zeros_like(fft2, dtype=torch.bool)
  mask[:, :, idx[:,0], idx[:,1]] = 1

  # Filter FFT result 
  fft2_filtered = fft2.where(mask, torch.zeros_like(fft2))  

  # Take 2D inverse FFT 
  feature_maps_recon = torch.fft.ifft2(fft2_filtered, dim=(2, 3)).real

  return feature_maps_recon

def highpass_filter_feature_pytorch(feature_maps, k):

    batch_size, num_features, height, width = feature_maps.shape
    feature_maps = feature_maps.view(batch_size, num_features, -1)
    feature_maps_fft = torch.fft.fft(feature_maps, dim=2)
    freqs = torch.fft.fftfreq(feature_maps_fft.shape[-1])

    _, idx = torch.topk(freqs.abs(), k, largest=True)
    mask = torch.zeros_like(feature_maps_fft, dtype=torch.bool)
    mask[:, :, idx] = 1

    feature_maps_fft_fil = feature_maps_fft.where(mask, torch.zeros_like(feature_maps_fft))
    feature_maps_recon = torch.fft.ifft(feature_maps_fft_fil, dim=2).real 
    
    return feature_maps_recon.view(batch_size, num_features, height, width)


def low_pass_DFT_pytorch(feature_maps, B):
    _, _, height, width = feature_maps.shape

    # Perform 2D FFT on each feature map
    feature_maps_fft = torch.fft.fftn(feature_maps, dim=[2, 3])

    # Shift the zero-frequency component to the center of the spectrum
    feature_maps_fft_shifted = torch.fft.fftshift(feature_maps_fft, dim=[2, 3])

    # Create a centered mask
    mask = torch.zeros_like(feature_maps_fft_shifted, dtype=torch.bool)
    center_x, center_y = height // 2, width // 2
    mask[:, :, center_x - B // 2 : center_x + B // 2, center_y - B // 2 : center_y + B // 2] = 1

    # Filter out frequency components outside the center
    feature_maps_fft_shifted_fil = feature_maps_fft_shifted.where(mask, torch.zeros_like(feature_maps_fft_shifted))
    
    # Shift the zero-frequency component back to the corners of the spectrum
    feature_maps_fft_fil = torch.fft.ifftshift(feature_maps_fft_shifted_fil, dim=[2, 3])

    # Perform 2D inverse FFT on each feature map
    feature_maps_recon = torch.fft.ifftn(feature_maps_fft_fil, dim=[2, 3]).real

    return feature_maps_recon


def high_pass_DFT_pytorch(feature_maps, B):
    _, _, height, width = feature_maps.shape

    # Perform 2D FFT on each feature map
    feature_maps_fft = torch.fft.fftn(feature_maps, dim=[2, 3], norm="ortho")

    # Shift the zero-frequency component to the center of the spectrum
    feature_maps_fft_shifted = torch.fft.fftshift(feature_maps_fft, dim=[2, 3])

    # Create a centered mask
    mask = torch.ones_like(feature_maps_fft_shifted, dtype=torch.bool)
    center_x, center_y = height // 2, width // 2
    mask[:, :, center_x - B // 2 : center_x + B // 2, center_y - B // 2 : center_y + B // 2] = 0

    # Filter out frequency components inside the center
    feature_maps_fft_shifted_fil = feature_maps_fft_shifted.where(mask, torch.zeros_like(feature_maps_fft_shifted))
    
    # Shift the zero-frequency component back to the corners of the spectrum
    feature_maps_fft_fil = torch.fft.ifftshift(feature_maps_fft_shifted_fil, dim=[2, 3])

    # Perform 2D inverse FFT on each feature map
    feature_maps_recon = torch.fft.ifftn(feature_maps_fft_fil, dim=[2, 3], norm="ortho").real 

    return feature_maps_recon