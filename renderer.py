#-- coding: UTF-8 -- 
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from DrawBresenhamLine import BresenhamLine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
rasterize_lines_module = tf.load_op_library("kernels/rasterize_lines_op.so")
line_renderer_module = tf.load_op_library("kernels/line_renderer_grad.so")

initial_pic = np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, ],
[-0.821823239326, -0.334920316935, 1.0, 0.0, 0.0, ],
[-0.772015750408, -0.156296133995, 1.0, 0.0, 0.0, ],
[-0.896534383297, -0.0893120840192, 1.0, 0.0, 0.0, ],
[-1.09576416016, 0.0, 1.0, 0.0, 0.0, ],
[-0.896534383297, 0.156296133995, 1.0, 0.0, 0.0, ],
[-0.597689568996, 0.200952187181, 1.0, 0.0, 0.0, ],
[-0.547882080078, 0.334920316935, 1.0, 0.0, 0.0, ],
[-0.697304487228, 0.558200538158, 1.0, 0.0, 0.0, ],
[-0.373555988073, 0.424232393503, 1.0, 0.0, 0.0, ],
[-0.298844784498, 0.446560412645, 1.0, 0.0, 0.0, ],
[-0.448267191648, 1.02708888054, 1.0, 0.0, 0.0, ],
[-0.149422392249, 0.535872519016, 1.0, 0.0, 0.0, ],
[-0.0249037332833, 0.759152650833, 1.0, 0.0, 0.0, ],
[0.72220826149, 1.71925759315, 1.0, 0.0, 0.0, ],
[0.72220826149, 0.558200538158, 1.0, 0.0, 0.0, ],
[1.04595685005, 0.491216421127, 1.0, 0.0, 0.0, ],
[1.61874270439, 0.580528557301, 1.0, 0.0, 0.0, ],
[0.996149301529, 0.178624168038, 1.0, 0.0, 0.0, ],
[1.27009046078, 0.0, 1.0, 0.0, 0.0, ],
[0.473170936108, -0.178624168038, 1.0, 0.0, 0.0, ],
[0.821823239326, -0.468888431787, 1.0, 0.0, 0.0, ],
[0.72220826149, -0.669840633869, 1.0, 0.0, 0.0, ],
[0.448267191648, -0.669840633869, 1.0, 0.0, 0.0, ],
[0.273941040039, -1.09407305717, 1.0, 0.0, 0.0, ],
[0.0249037332833, -1.42899334431, 1.0, 0.0, 0.0, ],
[-0.249037325382, -1.09407305717, 1.0, 0.0, 0.0, ],
[-0.249037325382, -0.558200538158, 1.0, 0.0, 0.0, ],
[-0.448267191648, -0.692168653011, 1.0, 0.0, 0.0, ],
[-0.647497057915, -0.714496672153, 0.0, 1.0, 0.0, ],
[-0.846726894379, -0.111640103161, 1.0, 0.0, 0.0, ],
[0.921438157558, -0.714496672153, 1.0, 0.0, 0.0, ],
[1.02105307579, -0.62518453598, 1.0, 0.0, 0.0, ],
[0.0747111961246, 0.200952187181, 1.0, 0.0, 0.0, ],
[0.124518662691, 1.45132136345, 1.0, 0.0, 0.0, ],
[0.0, 0.893120825291, 1.0, 0.0, 0.0, ],
[-0.0747111961246, 0.0893120840192, 0.0, 1.0, 0.0, ],
[-7.44621610641, -0.893120825291, 1.0, 0.0, 0.0, ],
[0.0996149331331, -0.223280206323, 1.0, 0.0, 0.0, ],
[-0.0498074665666, -1.67460155487, 1.0, 0.0, 0.0, ],
[0.149422392249, -0.066984064877, 1.0, 0.0, 0.0, ],
[2.58998823166, 0.982432842255, 1.0, 0.0, 0.0, ],
[0.0, 0.111640103161, 0.0, 1.0, 0.0, ],
[-1.27009046078, 2.85798668861, 1.0, 0.0, 0.0, ],
[-0.0996149331331, -0.200952187181, 1.0, 0.0, 0.0, ],
[0.124518662691, -0.223280206323, 1.0, 0.0, 0.0, ],
[0.124518662691, -0.0893120840192, 1.0, 0.0, 0.0, ],
[0.249037325382, 0.267936259508, 1.0, 0.0, 0.0, ],
[-0.273941040039, 0.111640103161, 1.0, 0.0, 0.0, ],
[-0.0747111961246, -0.111640103161, 1.0, 0.0, 0.0, ],
[0.0498074665666, -0.111640103161, 1.0, 0.0, 0.0, ],
[0.249037325382, 0.0446560420096, 1.0, 0.0, 0.0, ],
[-0.149422392249, 0.0, 0.0, 1.0, 0.0, ],
[3.78536748886, 0.803808748722, 1.0, 0.0, 0.0, ],
[-0.124518662691, -0.0223280210048, 1.0, 0.0, 0.0, ],
[-0.0747111961246, -0.267936259508, 1.0, 0.0, 0.0, ],
[0.174326121807, -0.0223280210048, 1.0, 0.0, 0.0, ],
[0.124518662691, 0.066984064877, 1.0, 0.0, 0.0, ],
[-0.199229866266, 0.0893120840192, 1.0, 0.0, 0.0, ],
[-0.224133595824, -0.111640103161, 1.0, 0.0, 0.0, ],
[-0.0996149331331, -0.245608210564, 1.0, 0.0, 0.0, ],
[0.473170936108, -0.066984064877, 1.0, 0.0, 0.0, ],
[-0.0498074665666, 0.334920316935, 1.0, 0.0, 0.0, ],
[-0.174326121807, 0.0, 0.0, 1.0, 0.0, ],
[-2.61489200592, 1.07174503803, 1.0, 0.0, 0.0, ],
[0.224133595824, 0.31259226799, 1.0, 0.0, 0.0, ],
[0.622593283653, -0.379576385021, 1.0, 0.0, 0.0, ],
[-0.124518662691, -0.066984064877, 1.0, 0.0, 0.0, ],
[-0.896534383297, 0.0, 1.0, 0.0, 0.0, ],
[0.672400832176, 0.379576325417, 1.0, 0.0, 0.0, ],
[-0.0249037332833, -0.357248336077, 1.0, 0.0, 0.0, ],
[-0.124518662691, 0.066984064877, 1.0, 0.0, 0.0, ],
[0.273941040039, -0.0446560420096, 0.0, 1.0, 0.0, ],
[-0.124518662691, 0.357248336077, 1.0, 0.0, 0.0, ],
[0.224133595824, 0.669840633869, 1.0, 0.0, 0.0, ],
[0.149422392249, 0.178624168038, 1.0, 0.0, 0.0, ],
[0.87163066864, 0.178624153137, 1.0, 0.0, 0.0, ],
[0.522978425026, -0.156296133995, 1.0, 0.0, 0.0, ],
[0.298844784498, -0.334920316935, 1.0, 0.0, 0.0, ],
[0.0249037332833, -0.156296133995, 0.0, 1.0, 0.0, ],
[-2.21643233299, -0.580528557301, 1.0, 0.0, 0.0, ],
[0.0996149331331, 0.558200538158, 1.0, 0.0, 0.0, ],
[-0.149422392249, 0.245608210564, 1.0, 0.0, 0.0, ],
[-0.398459732533, 0.133968129754, 1.0, 0.0, 0.0, ],
[-0.747111976147, 0.0, 1.0, 0.0, 0.0, ],
[-0.124518662691, -0.066984064877, 1.0, 0.0, 0.0, ],
[-0.298844784498, -0.513544440269, 0.0, 1.0, 0.0, ],
[2.8639292717, -0.31259226799, 1.0, 0.0, 0.0, ],
[1.09576416016, 0.0223280210048, 1.0, 0.0, 0.0, ],
[1.09576416016, -0.111640103161, 1.0, 0.0, 0.0, ],
[2.36585474014, -0.446560412645, 1.0, 0.0, 0.0, ],
[0.547882080078, -0.0223280210048, 0.0, 1.0, 0.0, ],
[-4.85622787476, 1.20571315289, 1.0, 0.0, 0.0, ],
[1.0708605051, 0.133968129754, 1.0, 0.0, 0.0, ],
[2.39075827599, 0.893120825291, 1.0, 0.0, 0.0, ],
[1.17047548294, 0.357248336077, 1.0, 0.0, 0.0, ],
[0.149422392249, 0.133968129754, 0.0, 1.0, 0.0, ],
[-4.65699768066, -0.200952187181, 1.0, 0.0, 0.0, ],
[0.149422392249, -0.178624168038, 0.0, 1.0, 0.0, ],
[0.323748528957, -0.29026427865, 1.0, 0.0, 0.0, ],
[0.149422392249, -0.0223280210048, 1.0, 0.0, 0.0, ],
[1.56893527508, 0.736824691296, 1.0, 0.0, 0.0, ],
[2.11681723595, 1.20571315289, 0.0, 1.0, 0.0, ],
[-8.1435213089, -3.79576349258, 1.0, 0.0, 0.0, ],
[-0.398459732533, -0.491216421127, 1.0, 0.0, 0.0, ],
[-1.41951274872, -0.915448844433, 1.0, 0.0, 0.0, ],
[-2.66469955444, -1.51830530167, 0.0, 1.0, 0.0, ],
[3.86007857323, 3.0142827034, 1.0, 0.0, 0.0, ],
[-2.71450686455, -0.29026427865, 1.0, 0.0, 0.0, ],
[-1.19537913799, -0.0446560420096, 0.0, 1.0, 0.0, ],
[4.43286466599, 1.11640107632, 1.0, 0.0, 0.0, ],
[-0.224133595824, 0.133968129754, 1.0, 0.0, 0.0, ],
[-0.821823239326, 0.178624168038, 1.0, 0.0, 0.0, ],
[-1.46932029724, 0.491216421127, 1.0, 0.0, 0.0, ],
[-1.44441652298, 0.62518453598, 0.0, 1.0, 0.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ],
[0.0, 0.0, 0.0, 0.0, 1.0, ]]])

incre_pic = np.array([[-0.468407,-0.152065,1.0,0.0,0.0,],
[-0.646569,-0.0765186,1.0,0.0,0.0,],
[-0.595236,0.000635311,1.0,0.0,0.0,],
[-0.648315,0.0577352,1.0,0.0,0.0,],
[-0.689304,0.171387,1.0,0.0,0.0,],
[-0.772079,0.340774,1.0,0.0,0.0,],
[-0.716741,0.459026,1.0,0.0,0.0,],
[-0.397069,0.43571,1.0,0.0,0.0,],
[-0.535522,0.753383,1.0,0.0,0.0,],
[-0.300937,0.556691,1.0,0.0,0.0,],
[-0.214077,0.58234,1.0,0.0,0.0,],
[-0.11581,0.888259,1.0,0.0,0.0,],
[0.000220828,0.878326,1.0,0.0,0.0,],
[0.13342,0.546899,1.0,0.0,0.0,],
[0.300846,0.539248,1.0,0.0,0.0,],
[0.451138,0.468894,1.0,0.0,0.0,],
[0.598876,0.338719,1.0,0.0,0.0,],
[1.0225,0.310629,1.0,0.0,0.0,],
[1.05508,0.235909,1.0,0.0,0.0,],
[1.09793,-2.49445e-05,1.0,0.0,0.0,],
[0.920766,-0.254876,1.0,0.0,0.0,],
[0.545141,-0.383481,1.0,0.0,0.0,],
[0.505441,-0.498718,1.0,0.0,0.0,],
[0.378551,-0.591728,1.0,0.0,0.0,],
[0.249,-0.674228,1.0,0.0,0.0,],
[0.00330743,-1.0967,1.0,0.0,0.0,],
[-0.204633,-0.975172,1.0,0.0,0.0,],
[-0.413048,-0.906662,1.0,0.0,0.0,],
[-0.471382,-0.750972,1.0,0.0,0.0,],
[-0.389237,-0.428562,1.0,0.0,0.0,],
[-0.307365,0.247718,1.0,0.0,0.0,],
[0.459067,-0.602156,1.0,0.0,0.0,],
[0.79276,-0.707789,1.0,0.0,0.0,],
[-0.282978,1.67667,1.0,0.0,0.0,],
[-0.321289,1.45059,1.0,0.0,0.0,],
[-0.0899436,1.15016,0.0,1.0,0.0,],
[-0.107153,0.472628,0.0,1.0,0.0,],
[-6.3199,-1.16896,1.0,0.0,0.0,],
[-0.773158,-0.640073,1.0,0.0,0.0,],
[-0.615703,-0.689451,1.0,0.0,0.0,],
[-0.412698,-0.0390696,1.0,0.0,0.0,],
[0.444888,0.507284,0.0,1.0,0.0,],
[-0.380939,3.10487,1.0,0.0,0.0,],
[-0.113921,-0.0387247,1.0,0.0,0.0,],
[0.0106836,0.0802776,1.0,0.0,0.0,],
[0.225506,-0.118531,1.0,0.0,0.0,],
[0.123087,0.000324398,1.0,0.0,0.0,],
[0.109349,0.189248,1.0,0.0,0.0,],
[-0.181129,0.0685314,1.0,0.0,0.0,],
[0.0274085,-0.170023,1.0,0.0,0.0,],
[0.173204,-0.0529387,1.0,0.0,0.0,],
[-0.0908108,0.0973614,1.0,0.0,0.0,],
[3.31945,0.221115,1.0,0.0,0.0,],
[-0.0970078,-0.193831,1.0,0.0,0.0,],
[-0.0864211,-0.133812,1.0,0.0,0.0,],
[0.290157,-0.126691,1.0,0.0,0.0,],
[-0.0578198,0.112285,1.0,0.0,0.0,],
[0.00641549,-0.140328,1.0,0.0,0.0,],
[0.202426,0.00207435,1.0,0.0,0.0,],
[-0.127346,0.179039,1.0,0.0,0.0,],
[-0.197255,0.0733959,1.0,0.0,0.0,],
[-2.26483,1.72013,1.0,0.0,0.0,],
[0.0773822,0.136612,1.0,0.0,0.0,],
[0.081393,-0.005959,1.0,0.0,0.0,],
[0.088644,-0.0891213,1.0,0.0,0.0,],
[-0.0507574,0.224621,1.0,0.0,0.0,],
[0.0957046,-0.0475387,1.0,0.0,0.0,],
[-0.0421275,0.249398,1.0,0.0,0.0,],
[0.0188567,0.0638333,0.0,1.0,0.0,],
[-0.154543,0.59191,1.0,0.0,0.0,],
[0.0250825,0.44644,1.0,0.0,0.0,],
[0.126755,0.505821,1.0,0.0,0.0,],
[0.299616,0.184367,1.0,0.0,0.0,],
[0.424179,0.0185398,1.0,0.0,0.0,],
[0.374427,-0.167098,1.0,0.0,0.0,],
[0.273199,-0.23312,0.0,1.0,0.0,],
[0.0897773,-0.319083,0.0,1.0,0.0,],
[-1.80347,-0.0841274,1.0,0.0,0.0,],
[-0.114468,0.510523,1.0,0.0,0.0,],
[-0.02297,0.353494,1.0,0.0,0.0,],
[-0.427156,0.288874,1.0,0.0,0.0,],
[-0.581975,-0.0254469,1.0,0.0,0.0,],
[-0.473006,-0.241656,1.0,0.0,0.0,],
[-0.139268,-0.299279,0.0,1.0,0.0,],
[3.18306,-0.248065,1.0,0.0,0.0,],
[0.686668,-0.215859,1.0,0.0,0.0,],
[2.3187,-0.466007,1.0,0.0,0.0,],
[1.05353,-0.243648,0.0,1.0,0.0,],
[0.595768,-0.15659,0.0,1.0,0.0,],
[-3.91284,1.22983,1.0,0.0,0.0,],
[0.961223,0.226186,1.0,0.0,0.0,],
[1.61236,0.165354,1.0,0.0,0.0,],
[0.894726,0.296672,0.0,1.0,0.0,],
[0.334682,0.172505,0.0,1.0,0.0,],
[-4.05936,-0.972402,1.0,0.0,0.0,],
[1.36533,0.795075,1.0,0.0,0.0,],
[-3.43835,-1.83056,1.0,0.0,0.0,],
[1.2378,0.477336,1.0,0.0,0.0,],
[1.04679,0.553562,1.0,0.0,0.0,],
[0.885285,0.412197,1.0,0.0,0.0,],
[-7.02418,-3.50459,1.0,0.0,0.0,],
[-0.467546,-0.151226,1.0,0.0,0.0,],
[-1.68425,-1.41983,1.0,0.0,0.0,],
[-0.662237,-0.304281,1.0,0.0,0.0,],
[3.77311,2.8897,1.0,0.0,0.0,],
[-0.397642,0.00529602,1.0,0.0,0.0,],
[-0.959085,0.0123075,0.0,1.0,0.0,],
[3.86546,1.14769,1.0,0.0,0.0,],
[-0.517914,0.0968665,1.0,0.0,0.0,],
[-0.608142,0.205989,1.0,0.0,0.0,],
[-2.70838,0.842551,1.0,0.0,0.0,],
[-0.911777,0.506777,0.0,1.0,0.0,],
[4.8989,-2.91924,0.0,0.0,1.0,],
[4.76147,-2.23549,0.0,0.0,1.0,],
[5.26915,-1.94412,0.0,0.0,1.0,],
[5.02341,-1.64901,0.0,0.0,1.0,],
[4.6259,-1.09049,0.0,0.0,1.0,],
[4.56009,-0.517091,0.0,0.0,1.0,],
[4.37413,-0.167207,0.0,0.0,1.0,],
[-0.00445842,0.245454,0.0,0.0,1.0,],
[-0.00305194,0.124255,0.0,0.0,1.0,],
[-0.0020923,0.171628,0.0,0.0,1.0,],
[-0.00277399,0.190716,0.0,0.0,1.0,],
[3.20707,0.567804,0.0,0.0,1.0,],
[2.98816,0.752125,0.0,0.0,1.0,],
[2.84541,0.92826,0.0,0.0,1.0,],
[2.52536,1.17691,0.0,0.0,1.0,],
[2.43307,1.45584,0.0,0.0,1.0,],
[1.62834,-0.0555186,0.0,0.0,1.0,],
[0.0,0.0,0.0,0.0,0.0,]])

initial_png = np.array([[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,],
[255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,],
[255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,],
[255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,]])
def getBoundary(abs_x, abs_y):
    min_x = min(abs_x)
    max_x = max(abs_x)
    min_y = min(abs_y)
    max_y = max(abs_y)
    boundary_x = max_x - min_x
    boundary_y = max_y - min_y
    return max(boundary_x, boundary_y)


def getMSE(vertice_1, vertice_2):
    # This function needs to be altered due to the definition of the MSE loss for two lines
    p1_x = vertice_1[0]
    p1_y = vertice_1[1]
    p2_x = vertice_2[0]
    p2_y = vertice_2[1]
    mse_loss = tf.square(p1_x-p2_x) + tf.square(p1_y - p2_y)
    return mse_loss



def render_lines(input_seqs, increment_seqs):
    """ Rasterize the input line.
    Args: 
        input_seqs: A tensor of shape (-1, 3)/(-1, 5), which does not matter. 
        
    Return:
        df_dvertices: The final gradients of the input vertices. f represents the target function, 
        defined by MSE of the input and output images. This tensor has the same shape as input_seqs.
    """

    # Get the absolute x, y coordinates of the original graph and the generated graph
    vec_ori_x = input_seqs[0][:, 0]
    vec_ori_y = input_seqs[0][:, 1]
    abs_x_start = [0]
    abs_y_start = [0]
    abs_x_end = []
    abs_y_end = []
    increment_x = increment_seqs[:, 0]
    increment_y = increment_seqs[:, 1]
    abs_x_end.append(0 + increment_x[0])
    abs_y_end.append(0 + increment_y[0])
    for i in range(1, vec_ori_x.shape[0]):
        abs_x_start.append(abs_x_start[i-1] + vec_ori_x[i])
        abs_y_start.append(abs_y_start[i-1] + vec_ori_y[i])
        abs_x_end.append(abs_x_start[i] + increment_x[i])
        abs_y_end.append(abs_y_start[i] + increment_y[i])
    abs_x_start = np.array(abs_x_start)
    abs_x_end = np.array(abs_x_end)
    abs_y_start = np.array(abs_y_start)
    abs_y_end = np.array(abs_y_end)
    
    # Get the boundaries of the two pictures and take the maximum of them, convert it to 
    # the upper integer
    image_bd_ori = getBoundary(abs_x_start, abs_y_start)
    image_bd_incre = getBoundary(abs_x_end, abs_y_end)
    image_bd = max(image_bd_incre, image_bd_ori)
    times = 48/float(image_bd)
    
    image_bd_input = int(image_bd)*2 # The attribute input of rasterize_lines.cc

    graph_out = np.zeros([48, 48])

    """for i in range(len(abs_x_start)):
        max_x = max(max_x, abs_x_end[i])
        max_y = max(max_y, abs_y_end[i])
        x = int((abs_x_end[i]+image_bd)*10)
        y = int((abs_y_end[i]+image_bd)*10)
    
        graph_out[x][y] = 255
    graph_out = Image.fromarray(graph_out.astype(np.uint8))
    graph_out.show()"""
    min_x_start = min(abs_x_start)
    min_y_start = min(abs_y_start)
    min_x_end = min(abs_x_end)
    min_y_end = min(abs_y_end)
    for i in range(len(abs_x_start)):
        """new_ori_x = int(round((abs_x_start[i] + image_bd)))
        new_ori_y = int(round((abs_y_start[i] + image_bd)))
        new_incre_x = int(round((abs_x_end[i] + image_bd)))
        new_incre_y = int(round((abs_y_end[i] + image_bd)))"""
        new_ori_x = int(round((abs_x_start[i] - min_x_start)*times))
        new_ori_y = int(round((abs_y_start[i] - min_y_start)*times))
        new_incre_x = int(round((abs_x_end[i] - min_x_end)*times))
        new_incre_y = int(round((abs_y_end[i] - min_y_end)*times))
        #print (new_ori_x, new_ori_y, new_incre_x, new_incre_y)
        resTmp = BresenhamLine(new_ori_x, new_ori_y, new_incre_x, new_incre_y)
        for j in range (0, len(resTmp)/2, 2):
            graph_out[resTmp[j]][resTmp[j+1]] = 255
    graph_out = Image.fromarray(graph_out.astype(np.uint8))
    graph_out = graph_out.rotate(270)
    graph_out_array = np.array(graph_out)
    #print (graph_out_array)
    #graph_out.show()
    loss_count = 0
    for i in range(48):
        for j in range(48):
            if int(initial_png[i][j]) != int(graph_out_array[i][j]):
                loss_count += 1
    #print (loss_count)
    # Initialize df_dvertices, which has shape [vertice_num, 2]
    df_dvertices = []
    for i in range(input_seqs.shape[1]):
        df_dvertices.append([0.0, 0.0])

    # Start calculating the gradient of the original picture's vertices
    for i in range(len(abs_x_start)):
        # All adding an constant does not change the relative relationships between the pixels
        # Choosing the closest integer of the coordinates
        new_ori_x = int(round(abs_x_start[i] + image_bd))
        new_ori_y = int(round(abs_y_start[i] + image_bd))
        new_incre_x = int(round(abs_x_end[i] + image_bd))
        new_incre_y = int(round(abs_y_end[i] + image_bd))

        # vertice_1 is a (2, ) tensor with value (x1, y1),
        # vertice_2 is a (2, ) tensor with value (x2, y2), vertice_2 is the input + incerment. 
        # The line formed by vertice_1 and vertice_2 represents the line formed by the initial state and ending state. 

        vertice_1 = np.array([new_ori_x, new_ori_y])
        vertice_2 = np.array([new_incre_x, new_incre_y])
        dline_dv_params, pixel_count, line_output = rasterize_lines_module.rasterize_lines(vertice_1, vertice_2, image_bd_input, image_bd_input)
        dline_dv_params = tf.reshape(dline_dv_params, [10, 6])
        pixel_count = tf.reshape(pixel_count, [1, ])
        line_output = tf.reshape(line_output, [10, 2])
        df_dlines = getMSE(vertice_1, vertice_2)
        dline_dvertices = line_renderer_module.line_renderer_grad(dline_dv_params, pixel_count)
        dline_dvertices = tf.reshape(dline_dvertices, [4, ])
        df_dlines = tf.cast(df_dlines, dtype=np.float32)
        df_dvertices_tmp = tf.multiply(df_dlines, dline_dvertices)
        
        df_dvertices[i] += df_dvertices_tmp[0:2]
        
        #df_dvertices[i+1] += df_dvertices_tmp[2:4]
    return pixel_count
    

@tf.RegisterGradient('RenderLines')
def _line_renderer_grad(dline_dv_params, pixel_count):
    return 0


    