import sys
sys.path.insert(0, '../..')

from libs_common.kernel_visualisation import *
import models.dqn_noisy_resnet_a.src.model            as Model


input_shape     = (4, 96, 96) 
outputs_count   = 9
model = Model.Model(input_shape, outputs_count)

model.load("models/dqn_noisy_resnet_a/")


kernel_visualisation = KernelVisualisation(model.layers_features, input_shape)

#kernel_visualisation.solve_layers([1, 3, 4, 5, 7, 8], "./layers_images/")
kernel_visualisation.solve_layers([4, 5, 7, 8], "./layers_images/")