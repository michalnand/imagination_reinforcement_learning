import torch
import torch.nn as nn
import numpy

from PIL import Image

class InputModel(torch.nn.Module):
    def __init__(self, in_shape):
        super(InputModel, self).__init__()
        self.weight  = nn.Parameter(torch.rand(in_shape)*2.0 - 1.0)

    def forward(self):
        return self.weight


class KernelVisualisation:
    def __init__(self, layers, input_shape):
        self.layers         = layers
        self.input_shape    = input_shape


    def solve_network(self, saving_path):
        for layer in range(len(self.layers)-2):
            self.solve_layer(layer, saving_path)

    def solve_layers(self, layers, saving_path):
        for layer in layers:
            self.solve_layer(layer, saving_path)

    def solve_layer(self, layer_id, saving_path):
        height  = self.input_shape[1]
        width   = self.input_shape[2]

        y = self._forward_to_layer(torch.randn((1, ) + self.input_shape ), layer_id)
        kernels_count = y.shape[1]

        tiles_height, tiles_width = self._make_rectangle(kernels_count)

        result_image = Image.new("RGB", (width*tiles_width, height*tiles_height))

        print("processing layer ", layer_id, kernels_count)

        for th in range(tiles_height):
            for tw in range(tiles_width):
                kernel_idx = th*tiles_width + tw
                
                print("   processing kernel ", kernel_idx)

                x = self._solve(layer_id, kernel_idx)
                x_rgb = self._to_rgb(x)
                im = Image.fromarray(x_rgb, "RGB")

                result_image.paste(im, (tw*width, th*height))


        file_name = saving_path + "layer_" + str(layer_id) + ".png"
        result_image.save(file_name)

    def _solve(self, layer_id, kernel_id, epoch_count=20, learning_rate = 0.1, decay = 0.00001, normalise = True):

        input_model = InputModel((1, ) + self.input_shape )
        optimizer   = torch.optim.Adam(input_model.parameters(), lr=learning_rate, weight_decay=decay)

        for epoch in range(epoch_count):
            x = input_model.forward()
            y = self._forward_to_layer(x, layer_id)

            '''
            maximize response from choosen kernel
            ''' 
            loss = -y[0][kernel_id].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            #print(layer_id, kernel_id, epoch, loss.detach().to("cpu").numpy())

        x_np = input_model.forward().detach().to("cpu").numpy()[0]

        if normalise:
            max = numpy.max(x_np)
            min = numpy.min(x_np)

            x_np = (x_np - min)*1.0/(max - min) 

        return x_np

    def _forward_to_layer(self, x_in, layer_id):
        y = x_in
        for i in range(layer_id + 1):
            y = self.layers[i](y)

        return y

    def _to_rgb(self, x):
        tmp = numpy.zeros((3, x.shape[1], x.shape[2]))
 
        tmp[0] = x[0]
        tmp[1] = x[1]
        tmp[2] = x[2]

        result = numpy.ascontiguousarray(tmp.transpose(1,2,0))
        result = numpy.array(result*255, dtype=numpy.uint8)

        return result


    def _make_rectangle(self, kernels_count):
        width   = int(kernels_count**0.5)
        height  = width

        while kernels_count%width != 0:
            width+= 1

        height = kernels_count//width

        return height, width 

