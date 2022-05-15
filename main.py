#!/usr/bin/python

import sys, getopt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train import trainModel
from plotsave import plotROC_outcomes
import argparse

def main(args):
    if not args.inputfile:
        print('Provide the input file: -i')
        sys.exit(2)
    if not args.outputfile:
        print('saving to the default')
        outputfile = 'ROC.png'
    else:
        outputfile = args.outputfile
    data = pd.read_csv(args.inputfile)
    true_variables, pred_variables, outcomes,models = trainModel(data)
    plotROC_outcomes(true_variables, pred_variables, outcomes, outputfile)

if __name__ == "__main__":
    #print(f"Arguments count: {len(sys.argv)}")
    #for i, arg in enumerate(sys.argv):
    #    print(f"Argument {i:>6}: {arg}")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile')
    parser.add_argument('-o','--outputfile')
    args = parser.parse_args()
    main(args)

    
