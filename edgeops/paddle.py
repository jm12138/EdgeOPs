import paddle
import paddle.nn as nn


class EdgeOP(nn.Layer):
    def __init__(self, kernel):
        '''
        kernel: shape(out_channels, in_channels, h, w)
        '''
        super(EdgeOP, self).__init__()
        out_channels, in_channels, h, w = kernel.shape
        self.filter = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(h, w),
            padding='SAME',
            bias_attr=False
        )
        self.filter.weight.set_value(kernel.astype('float32'))

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
            results = paddle.sum(paddle.abs(outputs), axis=1)
        elif mode == 1:
            results = paddle.sqrt(paddle.sum(paddle.pow(outputs, 2), axis=1))
        elif mode == 2:
            results = paddle.max(paddle.abs(outputs), axis=1)
        elif mode == 3:
            if weight is None:
                C = outputs.shape[1]
                weight = paddle.to_tensor([1/C] * C, dtype='float32')
            else:
                weight = paddle.to_tensor(weight, dtype='float32')
            results = paddle.einsum('nchw, c -> nhw', paddle.abs(outputs), weight)
        elif mode == 4:
            results = paddle.abs(outputs)
        return paddle.clip(results, 0, 255).cast('uint8')

    @paddle.no_grad()
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
