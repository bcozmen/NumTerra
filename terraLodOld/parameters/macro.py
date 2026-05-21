macro_1 = {
    'octaves' : 6, 
    'persistence': 0.49,
    'lacunarity': 2.02,
    'base_freq': 4.0,       # very low -> large blobs
    'warp_x': 0.42,        # no warp on macro layer -- pure continental blobs
    'warp_y': 0.23,
    'weight': 1.0,        # fraction of final height driven by macro layer
    'ridged': False,   # no ridges on macro layer -- pure FBM hills
}

macro_2 = {
    'octaves' : 6, 
    'persistence': 0.49,
    'lacunarity': 2.02,
    'base_freq': 5.0,       # very low -> large blobs
    'warp_x': 0.15,        # no warp on macro layer -- pure continental blobs
    'warp_y': 0.4,
    'weight': 0.75,        # fraction of final height driven by macro layer
    'ridged': False,   # no ridges on macro layer -- pure FBM hills
}

macro_3 = {
    'octaves' : 6, 
    'persistence': 0.49,
    'lacunarity': 2.02,
    'base_freq': 6.5,       # very low -> large blobs
    'warp_x': 0.2,        # no warp on macro layer -- pure continental blobs
    'warp_y': 0.08,
    'weight': 0.55,        # fraction of final height driven by macro layer
    'ridged': False,   # no ridges on macro layer -- pure FBM hills
}

macro_4 = {
    'octaves' : 6, 
    'persistence': 0.49,
    'lacunarity': 2.02,
    'base_freq': 7.0,       # very low -> large blobs
    'warp_x': 0.05,        # no warp on macro layer -- pure continental blobs
    'warp_y': 0.2,
    'weight': 0.4,        # fraction of final height driven by macro layer
    'ridged': False,   # no ridges on macro layer -- pure FBM hills
}

macro_params = [macro_1, macro_2, macro_3, macro_4]