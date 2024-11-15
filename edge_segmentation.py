from numpy import arange
import numpy as np
from pyvista import PolyData
import pyvista
import fast_simplification as fs
import time
from vtk import vtkPolyDataEdgeConnectivityFilter, vtkCellDataToPointData
import random
import scipy.linalg
import scipy.cluster
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from collections import Counter
inizio = time.time()

p = pyvista.Plotter(shape=(2,3) )
print("mesh_segmentation: Loading mesh...")

mesh0 = pyvista.read('prova.stl')
p.subplot(0,0)
p.add_mesh(mesh0, color='yellow')
p.view_xz()



def simplify(mesh):
    print("mesh_segmentation: Simplifing mesh...")
    
    points, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
    _, _, collapses = fs.simplify(points, faces,target_count=30000, return_collapses = True)
    _, _, indice_mapping = fs.replay_simplification(points,faces,collapses)
    mesh_out = fs.simplify_mesh(mesh,target_count=30000)
    return mesh_out, indice_mapping
def cluster(mesh,k):
    
    

    mesh.point_data['orig_point_idx'] = arange(mesh.n_points)
    feature_edges = mesh.extract_feature_edges(
        feature_angle=17,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_edges=True,
    )

    # There's some unfortunate API impedance mismatch here, vtkPolyDataEdgeConnectivityFilter wants a barrier edges 
    # specified with the same point indices as the original mesh, but extract_feature_edges only keeps points 
    # belonging to the feature edges
    edges = feature_edges.lines.reshape(-1, 3)
    edges[:, 1:] = feature_edges.point_data['orig_point_idx'][edges[:, 1:]]

    print("mesh_segmentation: Clustering...")
    feature_edges = PolyData(mesh.points, lines=edges.reshape(-1))
    # feature_edges.plot(show_edges = True)
    f = vtkPolyDataEdgeConnectivityFilter()
    f.SetExtractionModeToAllRegions()
    f.SetColorRegions(True)
    f.SetInputData(mesh)
    f.BarrierEdgesOn()
    f.SetBarrierEdges(True)
    f.SetSourceData(feature_edges)
    f.Update()
    mesh2 = PolyData(f.GetOutput())
    mesh2.point_data['orig_idx']= mesh['orig_point_idx']

    colors = np.array([[random.random(), random.random(), random.random()] for _ in range(len(mesh2['RegionId']))])
    mesh2['colors'] = colors[mesh2['RegionId']]
    p.subplot(0,2)
    p.add_title('OverSegmentation', font= 'courier', color= 'k', font_size=9)
    p.add_mesh(mesh2,scalars='colors',rgb=True, show_scalar_bar = False)
    
    p.view_xz()

#    p.screenshot('overseg.png', transparent_background=True)

    f.SetLargeRegionThreshold(0.01)
    f.GrowSmallRegionsOn()
    f.Update()

    mesh2 = PolyData(f.GetOutput())

    colors = np.array([[random.random(), random.random(), random.random()] for _ in range(len(mesh2['RegionId']))])
    mesh2['colors'] = colors[mesh2['RegionId']]

    def newborders(mesh):
        edges = []
        for i in range(mesh.n_cells):
            for j in range(3):
                if mesh['RegionId'][i] != mesh['RegionId'][mesh.cell_neighbors(i,'edges')[j]]:
                    edge = np.intersect1d(mesh.get_cell(i).point_ids, mesh.get_cell(mesh.cell_neighbors(i,'edges')[j]).point_ids)
                    edges.append([2 , edge[0], edge[1]])
        return np.array(edges)
    
    newborders= newborders(mesh2)
    smallgrow = PolyData(mesh2.points, lines=newborders.reshape(-1))


    p.subplot(1,0)
    p.add_title('SmallGrow', font= 'courier', color= 'k', font_size=9)
    p.add_mesh(mesh2,scalars='colors',rgb = True, show_scalar_bar = False)
    
    p.view_xz()

    
    v = vtkPolyDataEdgeConnectivityFilter()
    v.SetExtractionModeToAllRegions()
    v.SetColorRegions(True)
    v.SetInputData(mesh2)
    v.BarrierEdgesOn()
    v.SetBarrierEdges(True)
    v.SetSourceData(smallgrow)
    v.SetLargeRegionThreshold(0.02)
    v.GrowLargeRegionsOn()
    
    v.Update()
    number_of_extracted_regions = v.GetNumberOfExtractedRegions()


    mesh2 = PolyData(v.GetOutput())


    colors = np.array([[random.random(), random.random(), random.random()] for _ in range(len(mesh2['RegionId']))])
    mesh2['colors'] = colors[mesh2['RegionId']]

    p.subplot(1,1)
    p.add_title('LargeGrow', font= 'courier', color= 'k', font_size=9)
    p.add_mesh(mesh2,scalars='colors', show_scalar_bar = False)
    
    p.view_xz()

    mesh2['Regions'] = mesh2['RegionId'][mesh2['orig_point_idx']]

#    p.screenshot('final.png', transparent_background=True)
    

    return mesh2, number_of_extracted_regions


#Semplifico la mesh e la stampo
mesh_out, indice_mapping = simplify(mesh0)
p.subplot(0,1)
p.add_mesh(mesh_out,  color='yellow')
p.view_xz()
#Lancio il cluster
mesh0_cluster, number_of_extracted_regions = cluster(mesh_out,0)

#Transposing back on the original Mesh
maria = mesh0_cluster['RegionId']
RegionId = np.zeros(mesh0_cluster.n_points)
for i in range(mesh0_cluster.n_points):
    adjacente_cells = mesh_out.point_cell_ids(i)
    RegionId[i] = maria[adjacente_cells[0]]
RegionId_final=RegionId[indice_mapping]

mesh0['RegionId'] = RegionId_final
# mesh0 = mesh0.point_data_to_cell_data()
Region_Id_cells = []
for i in range(mesh0.n_cells):
    counter = Counter(mesh0['RegionId'][mesh0.get_cell(i).point_ids])
    Region_Id_cells.append(counter.most_common(1)[0][0])
Region_Id_cells = np.array(Region_Id_cells)

#Printing
colors = np.array([[random.random(), random.random(), random.random()] for _ in range(len(Region_Id_cells))])
mesh0['colors'] = colors[Region_Id_cells.astype(int)]
p.subplot(1,2)
p.add_mesh(mesh0,scalars='colors',rgb = True, show_scalar_bar = False)
p.view_xz()



cluster_red = []
for i in range(number_of_extracted_regions):
    cluster_red.append(mesh0.extract_cells(mesh0.cell_data['RegionId'] == i))

for i in range(number_of_extracted_regions):
    cluster_red[i].extract_surface().save(f'exp/region_{i}.stl')

mesh0_cluster.save('export_prova.stl')


fine = time.time()
minuti = (fine - inizio)//60
secondi = (fine - inizio)%60
print(f'mesh_segmentation: Total time {int(minuti)} and {int(secondi)} s')



p.show()

