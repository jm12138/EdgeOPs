import torch
import torch.nn as nn


class EdgeOP(nn.Module):
    def __init__(self, kernel):
        '''
        kernel: shape(out_channels, in_channels, h, w)
        '''
        super(EdgeOP, self).__init__()
        out_channels, in_channels, h, w = kernel.shape
        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(h, w),
            padding='same',
            bias=False
        )
        self.filter.weight.data.copy_(torch.from_numpy(kernel.astype('float32')))

    @staticmethod
    def postprocess(outputs, mode=0, weight=None):
        '''
        Input: NCHW
        Output: NHW(mode==1-3) or NCHW(mode==4)

        Params:
            mode: switch output mode(0-4)
            weight: weight when mode==3
        '''
        device = outputs.device
        if mode == 0:
            results = torch.sum(torch.abs(outputs), dim=1)
        elif mode == 1:
            results = torch.sqrt(torch.sum(torch.pow(outputs, 2), dim=1))
        elif mode == 2:
            results = torch.max(torch.abs(outputs), dim=1)
        elif mode == 3:
            if weight is None:
                C = outputs.shape[1]
                weight = torch.from_numpy([1/C] * C, dtype=torch.float32).to(device)
            else:
                weight = torch.from_numpy(weight, dtype=torch.float32).to(device)
            results = torch.einsum('nchw, c -> nhw', torch.abs(outputs), weight)
        elif mode == 4:
            results = torch.abs(outputs)
        return torch.clip(results, 0, 255).byte()

    @torch.no_grad()
    def forward(self, images, mode=0, weight=None):
        '''
        Input: NCHW
        Output: NHW(mode==1-3) or NCHW(mode==4)

        Params:
            images: input tensor of images
            mode: switch output mode(0-4)
            weight: weight when mode==3
        '''
        outputs = self.filter(images)
        return self.postprocess(outputs, mode, weight)


if __name__=='__main__':
    import numpy as np
    def Roberts():
        kernel = np.array([
            [[
                [1,  0],
                [0, -1]
            ]],
            [[
                [0, -1],
                [1,  0]
            ]]
        ])
        return EdgeOP(kernel)

    Roberts()(torch.randn(1, 1, 512, 512))