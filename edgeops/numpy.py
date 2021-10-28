import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), kernel=None):
        '''
        kernel: shape(out_channels, in_channels, h, w)
        '''
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding == 'same':
            assert stride==(1, 1), 'No support stride more than 1.'
            self.padding = (kernel_size[0]//2, kernel_size[1]//2-((kernel_size[1]-1)%2))
        if kernel is None:
            self.kernel = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        else:
            self.kernel = kernel

    @staticmethod
    def im2col(input, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), writeable=False):
        if padding != (0, 0):
            assert not writeable, 'No writable in padding mode.'
            input = np.pad(input, ((0, 0), (0, 0), padding, padding), mode='constant')
        isize = input.shape
        istrides = input.strides

        H = (isize[2]-(dilation[0]*(kernel_size[0]-1)+1))/(stride[0])+1
        W = (isize[3]-(dilation[1]*(kernel_size[1]-1)+1))/(stride[1])+1
        assert int(H) == H and int(W) == W, 'conv2d not aligned'
        H = int(H)
        W = int(W)
        istrides = list(istrides+istrides[-2:])
        istrides[2] *= stride[0]
        istrides[3] *= stride[1]
        istrides[4] *= dilation[0]
        istrides[5] *= dilation[1]
        return np.lib.stride_tricks.as_strided(input,
                                            (isize[0], isize[1], H,
                                                W, kernel_size[0], kernel_size[1]),
                                            istrides,
                                            writeable=writeable,
                                            )

    def __call__(self, x):
        x = self.im2col(x, self.kernel_size, self.stride, self.padding, self.dilation)
        return np.tensordot(x, self.kernel, [(1, 4, 5), (1, 2, 3)]).transpose(0, 3, 1, 2)

class EdgeOP:
    def __init__(self, kernel):
        '''
        kernel: shape(out_channels, in_channels, h, w)
        '''
        super(EdgeOP, self).__init__()
        out_channels, in_channels, h, w = kernel.shape
        self.filter = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(h, w),
            padding='same',
            kernel=kernel.astype('float32')
        )

    @staticmethod
    def postprocess(outputs, mode=0, weight=None):
        '''
        Input: NCHW
        Output: NHW(mode==1-3) or NCHW(mode==4)

        Params:
            mode: switch output mode(0-4)
            weight: weight when mode==3
        '''
        if mode == 0:
            results = np.sum(np.abs(outputs), axis=1)
        elif mode == 1:
            results = np.sqrt(np.sum(np.pow(outputs, 2), axis=1))
        elif mode == 2:
            results = np.max(np.abs(outputs), axis=1)
        elif mode == 3:
            if weight is None:
                C = outputs.shape[1]
                weight = np.array([1/C] * C, dtype=np.float32)
            else:
                weight = np.array(weight, dtype=np.float32)
            results = np.einsum('nchw, c -> nhw', np.abs(outputs), weight)
        elif mode == 4:
            results = np.abs(outputs)
        return np.clip(results, 0, 255).astype('uint8')

    def __call__(self, images, mode=0, weight=None):
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

    Roberts()(np.random.randn(1, 1, 512, 512))