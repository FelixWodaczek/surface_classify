#!/usr/bin/env python3.9

import sys
sys.path.append("py_src/")

import sort_neigh as sn
import time

import numpy as np

from argparse import ArgumentError, ArgumentParser

class SNParser:
    def __init__(self):
        self.parser = ArgumentParser(
            description="""Sort the surface types of 'Rh' in  a nanoparticle provided as a .lammpstrj over timesteps by its nearest neighbours.
            A directory is created containing a subdirectory for each of the timesteps the nanoparticle is given over.
            The number of timesteps and the number of atoms in the nanoparticle need to be properly set (see --help).
            Furthermore, sort_neigh assumes that the atoms of interest sit in the last n places over all timesteps of nanoparticle evolution.
            If there are more or less atoms than 14, set --last_n accordingly.
            
            The name of the directory to place the results in can be specified.
            If none is provided a standard output directory is created next to the input file.
            
            See --help on how to set the parameters for SOAP or the kernel which classifies the surfaces.
            
            The program returns the state of all 'Rh' atoms for each timestep in a matrix as a .txt file."""
        )
        self.neighbour_sort = None
        self.pars_args = None

    def init_parser(self):
        self.parser.add_argument(
            dest="func", type=str,
            help="""Subfunction of sort_neighbour.py to call.
            Available Functions:
            sort_cat - Sorts Nanoparticle in file according to provided arguments.
            plot_result - Plot result originating from sort_cat in .txt file according to arguments.
            """
        )
        self.parser.add_argument(
            dest="file", type=str, 
            help="Name of file containing nanoparticle."
        )
        self.parser.add_argument(
            "--timesteps",
            dest="n_ts", default=100, type=int,
            help="Number of timesteps for nanoparticle. Defaults to 100."
        )
        self.parser.add_argument(
            "--num_atoms",
            dest="n_ats", default=1400, type=int,
            help="Number of atoms in nanoparticle. Defaults to 1400"
        )
        self.parser.add_argument(
            "--last_n",
            dest="last_n", default=14, type=int, 
            help="The last n atoms in the nanoparticles to analyze using sort_neigh."
        )
        self.parser.add_argument(
            "-o", "--output_directory",
            dest="out_dir", default=None,
            help="""Output directory in which to place analysed timesteps. 
            If none is provided, defaults to 'neigh_sort_out'."""
        )
        self.parser.add_argument(
            "-O", "--output_file",
            dest="out_file", default=None,
            help="""Output file in which to place analysed timesteps. 
            If none is provided, defaults to 'part_count.txt'."""
        )
        self.parser.add_argument(
            "--r_cut",
            dest="rcut", default=3.1, type=float,
            help="rcut in SOAP. Defaults to 3.1."
        )
        self.parser.add_argument(
            "--n_max",
            dest="nmax", default=12, type=int,
            help="nmax in SOAP. Defaults to 12."
        )
        self.parser.add_argument(
            "--l_max",
            dest="lmax", default=12, type=int,
            help="lmax in SOAP. Defaults to 12."
        )
        self.parser.add_argument(
            "--sigma",
            dest="sigma", default=0.5, type=float,
            help="lmax in SOAP. Defaults to 0.5."
        )
        self.parser.add_argument(
            "--gamma_kernel",
            dest="gamma_kernel", default=0.05, type=float,
            help="gamma_kernel in dscribe AverageKernel. Defaults to 0.05."
        )
        self.parser.add_argument(
            "--create_subfolders",
            dest="create_subfolders", default=1,
            help="""Whether to create subfolders within ts folders for each surface type. 
            Set to 0 for False and to any integer for True. 
            Defaults to 1."""
        )
        self.parser.add_argument(
            "--classify_mode",
            dest="classify_mode", default="pre_group", type=str, 
            help="""Mode by which to classify the surface.
            Available modes: "pre_group", "class_all".
            "pre_group" only compares to atoms that match number of nearest neighbours.
            "class_all" compares to the full kernel and returns neighbour numbers and classes according to closest category.
            Defaults to pre_group."""
        )

    def read_args(self):
        self.pars_args = self.parser.parse_args()

        if self.pars_args.out_file is not None:
            self.pars_args.out_file = str(self.pars_args.out_file)

        if self.pars_args.out_dir is not None:
            self.pars_args.out_dir = str(self.pars_args.out_dir)
    
    def init_neighbour_sort(self):
        self.neighbour_sort = sn.NeighbourSort(
        rcut=self.pars_args.rcut, nmax=self.pars_args.nmax, lmax=self.pars_args.lmax,
        sigma=self.pars_args.sigma, gamma_kernel=self.pars_args.gamma_kernel
        )

    def sort_cat(self):
        start = time.time()
        self.neighbour_sort.init_folder_structure(
            self.pars_args.file, n_atoms_in_part=self.pars_args.n_ats,
            timesteps=int(self.pars_args.n_ts), out_dir=self.pars_args.out_dir
        )
        cat_counter = self.neighbour_sort.create_local_structure(
            last_n=self.pars_args.last_n, create_subfolders=self.pars_args.create_subfolders,
            mode=self.pars_args.classify_mode
        )
        stop = time.time()
        print("Sorting took %u seconds."%(stop-start))
        file_name = self.neighbour_sort.sort_save_cat(
            file_name = self.pars_args.out_file, cat_counter=cat_counter
        )
        print("Saved output to '%s'."%file_name)

    def plot_result(self):
        sorted_cat_counter, timesteps, sorted_classes = self.neighbour_sort.load_sort_cat(file_name=self.pars_args.file)
        self.neighbour_sort.plot_dist(sorted_classes, sorted_cat_counter)
        

def main():
    snp = SNParser()
    snp.init_parser()
    snp.read_args()
    
    snp.init_neighbour_sort()

    if str(snp.pars_args.func) == "sort_cat":
        snp.sort_cat()

    elif str(snp.pars_args.func) == "plot_result":
        snp.plot_result()

    else:
        raise ValueError("Function %s not implemented. See --help for available functions in sort_neigh_terminal."%str(snp.pars_args.func))

    
if __name__ == '__main__':
    main()

    # TODO: classify single particle over time through classes
    # TODO: find better way to classify?