import os
import pickle
import datamapplot
import numpy as np
import pandas as pd

data_dir = '/Volumes/MDatahubDev/Total_result_integration/integrated'

coords = np.load(os.path.join(data_dir, 'methods_coords.npy'))
coords.shape

cluster_labels_file = os.path.join(data_dir, 'methods_labels.pkl')
cluster_labels = pickle.load(open(cluster_labels_file, 'rb'))

texts_file = os.path.join(data_dir, 'methods_texts.pkl')
texts = pickle.load(open(texts_file, 'rb'))

plot = datamapplot.create_interactive_plot(
    coords,
    cluster_labels,
    hover_text = texts,
    point_radius_max_pixels=6,
    width=1000,
    height=1000,
    cluster_boundary_polygons=True,
    cluster_boundary_line_width=6,
    title="Human Gut Microbiome Experimental Methods",
    sub_title="mdatahub projects to extract sample information from thousands of papers",
    enable_search=True,
    darkmode=True,
)

plot.save("EMBERS_Methods.html")