import pygame
from track_generator import *

width       = 500
height      = 400
line_width  = 10

track = TrackPlaneGenerator(width=20, width_min=line_width, height_min=line_width, width_max=width-line_width, height_max=height-line_width)

#Set up pygame
pygame.init()

#Set up the window
windowSurface = pygame.display.set_mode((width, height), 0 , 32)
pygame.display.set_caption("car race")


#pygame.draw.polygon(windowSurface, (255, 255, 255), ((250, 0), (500,200),(250,400), (0,200) ))

pygame.draw.lines(windowSurface, (255, 255, 255), False, track.points, width=line_width)


pygame.display.update()



while True:
    pass