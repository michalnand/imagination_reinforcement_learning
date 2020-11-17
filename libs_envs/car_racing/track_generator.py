import xml.etree.ElementTree as ET
import os
import numpy
import json

import matplotlib.pyplot as plt


class TrackPlaneGenerator():
    
    def __init__(self, base_points_count = 1024, width = 15.0, width_min = -1.0, height_min = -1.0, width_max = 1.0, height_max = 1.0):

        self.base_points_count  = base_points_count
        self.width              = width/1000.0
        self.decimal_places     = 7

        points = []


        p_curve_change = 0.3 #0.01 .. 0.2
        dr_max         = 0.04


        dphi         = 2.0*numpy.pi/self.base_points_count
        if numpy.random.randint(2) == 1:
            dphi = -dphi
 
        line_curve_mode = "right"

        r_max       = 2.0
        r_initial   = r_max/2
        r           = r_initial
        dr          = 0.0
        dr_smooth   = 0.0
        
        phi = 0.0
        k   = 0.02

        for i in range(self.base_points_count):
            
            t = numpy.exp(10*(i/self.base_points_count - 1.0))
            rw = t*r_initial + (1.0 - t)*r

            x = rw*numpy.sin(phi)
            y = rw*numpy.cos(phi)


            if numpy.random.rand() < p_curve_change:
                rnd = numpy.random.randint(2)
                if rnd == 0:
                    line_curve_mode = "right"
                    dr = numpy.random.rand()*dr_max + 0.001
                elif rnd == 1:
                    line_curve_mode = "left"
                    dr = -(numpy.random.rand()*dr_max + 0.001)
                else:
                    line_curve_mode = "sraight"
                    dr = self._calc_dr_for_straight(r, dphi, dphi)

                    

            dr_smooth = (1.0 - k)*dr_smooth + k*dr
            r = numpy.clip(r + dr_smooth, 0.1*r_max, r_max)
            

            phi+= dphi

            points.append([x, y])

        self.points = numpy.transpose(numpy.asarray(points))

        max = numpy.max(self.points, axis = 1)
        min = numpy.min(self.points, axis = 1)

        range_values_min = numpy.asarray([width_min, height_min])
        range_values_max = numpy.asarray([width_max, height_max])

        k = (range_values_max - range_values_min)/(max - min)
        q = range_values_min - k*min

        self.points[0] = self.points[0]*k[0] + q[0]
        self.points[1] = self.points[1]*k[1] + q[1]

        self.points = numpy.transpose(self.points)

    def show(self, file_name = None):
        
        tmp = numpy.transpose(self.points)

        plt.clf()
        plt.plot(tmp[0], tmp[1])

        if file_name is not None:
            plt.savefig(file_name)
        else:
            plt.show()

    def save(self, idx, path = "./models_tracks/"):

        color_idx = numpy.random.randint(2)

        if color_idx == 0:
            plane_color = [1, 1, 1, 1]
            line_color  = [0, 0, 0, 1]
        elif color_idx == 1:
            plane_color = [0, 0, 0, 1]
            line_color  = [1, 1, 1, 1]

        file_name_prefix = path + str(idx)

        self._save_obj(file_name_prefix)
        self._save_urdf(plane_color, line_color, file_name_prefix)
        self._save_json(file_name_prefix)
        self.show(file_name_prefix + ".png")

   
    

if __name__ == "__main__":
    generator = TrackPlaneGenerator(1024, 15.0)
    generator.show()
