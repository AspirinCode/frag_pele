import argparse
import sys
import os
import glob
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from cluster_drug_discovery.methods import kmeans, dbscan, hdbscan, agglomerative 
from cluster_drug_discovery.input_preprocess import coords_extract as ce
import cluster_drug_discovery.visualization.plots as pl
from sklearn import cluster, datasets, mixture
from AdaptivePELE.utilities import utilities
from AdaptivePELE.utilities import utilities
from AdaptivePELE.analysis import splitTrajectory, simulationToCsv 
from AdaptivePELE.analysis import splitTrajectory, simulationToCsv 


def run(pele_sim, residue,  limit_structs=1, trajectory_basename="trajectory_", top=None, cpus=1, nclust=15):

    #Initialize variables
    traj_name = "*/{}*.pdb".format(trajectory_basename)
    pele_path = os.path.join(pele_sim, traj_name)
    pdb_files = glob.glob(pele_path)

    #Extract ligand coordinates
    feat_ext = ce.CoordinatesExtractor(pdb_files, [residue, ], cpus)
    if os.path.exists("extracted_feature.txt"):
        X = np.loadtxt("extracted_feature.txt") 
    else:
        X, samples = feat_ext.retrieve_coords()
        np.savetxt("extracted_feature.txt", X)
    
    #Clusterize
    cluster = agglomerative.AgglomerativeAlg(X, nclust) 
    y_pred = cluster.run()

    
    silhouette_values = silhouette_samples(X, y_pred)
    idx = np.argsort(silhouette_values)[::-1]

    #Check that there is no previously generated file
    try:
        samples[idx]
    except NameError:
        raise NameError("Remove previously generated extracted_feature.txt file and run again") 

    #Extract cluster
    for i in range(0, cluster.nclust):
        output_structs = 0
        for clust, sample, silh in zip(y_pred[idx], samples[idx], silhouette_values[idx]):
            if output_structs < limit_structs and clust == i:
                output_structs += 1
                if pele_path[-3:] == "pdb":
                    topology = None
                    filename = "path{}.{}.{}.cluster{}.pdb".format(sample.epoch, sample.traj, sample.model, i)
                    trajectory = os.path.join(pele_sim, "{}/{}{}.pdb".format(sample.epoch, trajectory_basename, sample.traj))
                    snapshots = utilities.getSnapshots(trajectory, topology=topology, use_pdb=False)
                    with open(filename, "w") as fw:
                        fw.write(snapshots[sample.model-1])
                elif pele_path[-3:] == "xtc":
                    topology = top
                    filename = "path{}.{}.{}.cluster{}.pdb".format(sample.epoch, sample.traj, sample.model, i)
                    trajectory = os.path.join(pele_sim, "{}/{}{}.xtc".format(sample.epoch, trajectory_basename, sample.traj))
                    splitTrajectory.main("", [trajectory, ], topology, [sample.model,],template=filename, use_pdb=False)

def add_args(parser):
    parser.add_argument('pele_sim', type=str, help="output of pele simulation")
    parser.add_argument('residue', type=str, help="residue of the ligand")
    parser.add_argument('--struct_per_cluster', type=int, help="Structure to retrieve per cluster", default=1)
    parser.add_argument('--traj_name', type=str, help="Name of the trajectory files. i.e trajectory_", default="trajectory_")
    parser.add_argument('--top', type=str, help="Topology file. Only necessary if using xtc", default="")
    parser.add_argument('--cpus', type=int, help="cpus", default=1)
    parser.add_argument('--nclust', type=int, help="Number of clusters. Defulat=15", default=15)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusterization algorithm with analysis techniques')
    parser = add_args(parser)
    args = parser.parse_args()
    cluster(args.pele_sim, args.residue,  args.struct_per_cluster, args.traj_name, args.top, args.cpus, args.nclust) 
