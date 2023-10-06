#!/bin/usr/env python

import argparse
import numpy as np
import yaml
from multiprocessing import Pool
import sys, os
from pyGSI.diags import Conventional
from pyGSI.plot_diags import plot_histogram
from datetime import datetime

###############################################

start_time = datetime.now()

# Parse command line
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--nprocs",
                help="Number of tasks/processors for multiprocessing")
ap.add_argument("-y", "--yaml",
                help="Path to yaml file with diag data")
ap.add_argument("-y2", "--yaml2",
                help="Path to 2nd yaml file with diag data")
ap.add_argument("-e", "--exp",
                help="Name of experiment")
ap.add_argument("-o", "--outdir",
                help="Out directory where files will be saved")
myargs = ap.parse_args()

input_yaml = myargs.yaml
input_yaml2 = myargs.yaml2
exp = myargs.exp
outdir = myargs.outdir

with open(input_yaml, 'r') as file:
    parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)
with open(input_yaml2, 'r') as file2:
    parsed_yaml_file2 = yaml.load(file2, Loader=yaml.FullLoader)

conv_config = (parsed_yaml_file['diagnostic'])		# o-f (ges)
conv_config2 = (parsed_yaml_file2['diagnostic'])	# o-a (anl)

###############################################

# Loop over entries in yaml files to create all plots.
# Only need to loop over one yaml file because all variables are the same 
# except for diagfile and diag_type.

i = 0	# Needed to obtain diagfile and diag_type from 2nd yaml file
for conv in conv_config:
    conv['outdir'] = outdir

# Get information from o-f file
    diagfile = conv['conventional input']['path']
    diag_type = conv['conventional input']['data type']  
    obsid = conv['conventional input']['observation id']
    analysis_use = conv['conventional input']['analysis use']
    plot_type = conv['conventional input']['plot type']

    diagfile = ''.join(diagfile)
    diag_type = ''.join(diag_type)
    diag_type = diag_type.lower()

# Get information from o-a file
    diagfile2 = conv_config2[i]['conventional input']['path']
    diag_type2 = conv_config2[i]['conventional input']['data type']

    diagfile2 = ''.join(diagfile2)
    diag_type2 = ''.join(diag_type2)
    diag_type2 = diag_type2.lower()

# Read in data and create histogram plot
    diag = Conventional(diagfile)
    diag2 = Conventional(diagfile2)

    diag_components = diagfile.split('/')[-1].split('.')[0].split('_')
    if diag_components[1] == 'conv' and diag_components[2] == 'uv':
        u, v = diag.get_data(diag_type, obsid=obsid,
                             analysis_use=analysis_use)
            
        data = {'assimilated': {'u': u['assimilated'],
                                'v': v['assimilated']
                               },
                'monitored':   {'u': u['monitored'],
                                'v': v['monitored']
                               }
               }

        u, v = diag2.get_data(diag_type2, obsid=obsid,
                             analysis_use=analysis_use)

        data2 = {'assimilated': {'u': u['assimilated'],
                                 'v': v['assimilated']
                               },
                'monitored':   {'u': u['monitored'],
                                'v': v['monitored']
                               }
               }

    else:
        data = diag.get_data(diag_type, obsid=obsid,
                             analysis_use=analysis_use)
            
        data = {'assimilated': data['assimilated'],
                'monitored': data['monitored']
               }

        data2 = diag2.get_data(diag_type2, obsid=obsid,
                             analysis_use=analysis_use)

        data2 = {'assimilated': data2['assimilated'],
                'monitored': data2['monitored']
               }
 
    metadata = diag.metadata
    metadata2 = diag2.metadata


# Need to read in both o-f and o-a data in order to set axes limits correctly
# Create o-f plot
    if np.isin('histogram', plot_type):
        plot_histogram(data, data2, metadata, exp, outdir)
# Create o-a plot
    if np.isin('histogram', plot_type):
        plot_histogram(data2, data, metadata2, exp, outdir)

    i += 1

print(datetime.now() - start_time)
