import torch

class PSNRLoss(torch.nn.Module):
    def __init__(self, PIXEL_MAX: float=255.0, inverse: bool=False) -> None:
        '''
        Initialize the PSNR loss function.
        Parameters:
            PIXEL_MAX: The maximum value of a pixel in the image.
            inverse: If true, the loss function will return the negative of the PSNR.
        '''
        super(PSNRLoss, self).__init__()
        self.BCELoss = torch.nn.BCELoss(reduction='mean')
        self.PIXEL_MAX = PIXEL_MAX
        self.inverse = inverse
        self.name ="Peak Signal-to-Noise Ratio (PSNR)"
    
    def forward(self, noisy_signal, true_signal, noisy_label=None, true_label=None):
        '''
        Forward pass of the PSNR loss function.
        Parameters:
            noisy_signal: The noisy signal.
            true_signal: The true signal.
            noisy_label: The noisy label.
            true_label: The true label.
        Returns:
            The PSNR loss.
        '''
        # Compute signal mse and signal psnr
        noisy_signal = torch.flatten(noisy_signal).type(torch.FloatTensor) / 255.0
        true_signal = torch.flatten(true_signal).type(torch.FloatTensor) / 255.0
        mse_loss = torch.mean((noisy_signal - true_signal)**2)
        if mse_loss == 0:
            signal_psnr = torch.scalar_tensor(100.0).type(torch.FloatTensor)
        else:
            signal_psnr = 20 * torch.log10(1.0 / mse_loss)

        # Compute label bce and psnr of labels are provided
        if noisy_label == None and true_label == None:
            bce_loss = torch.scalar_tensor(0.0).type(torch.FloatTensor)
        elif noisy_label != None and true_label != None:
            noisy_label = torch.flatten(noisy_label).type(torch.FloatTensor)
            true_label = torch.flatten(true_label).type(torch.FloatTensor)
            bce_loss = self.BCELoss(input=noisy_label, target=true_label)
        if bce_loss == 0:
            label_psnr = torch.scalar_tensor(100.0).type(torch.FloatTensor)
        else:
            label_psnr = 20 * torch.log10(1.0 / bce_loss)
        
        # Compute overall PSNR
        psnr = signal_psnr + label_psnr
        if self.inverse:
            psnr = -psnr
        return psnr