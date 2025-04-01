 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:09:39 2023

@author: luismoya
"""
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import matplotlib.pyplot as plt

def GetNodesFromSHP (inputShp, distThreshold= 5):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSourceLinks = driver.Open(inputShp, 0)
    layerLines = dataSourceLinks.GetLayer()
    nodesTmp= []
    for feature in layerLines:
        # print(feature)
        line= feature.GetGeometryRef()
        # print(line)
        num_points = line.GetPointCount()
        for i in range(num_points):
            x, y, z = line.GetPoint(i)
            # a= np.
            nodesTmp.append([x,y])
            print(x,y,z)
    nodesTmp= np.array(nodesTmp)
    print(nodesTmp.shape)
    nodes= [nodesTmp[0,:]]
    for i in range(1,nodesTmp.shape[0]):
        point= nodesTmp[i,:]
        dist= (nodesTmp[:i,0]-point[0])**2 + (nodesTmp[:i,1]-point[1])**2
        if not np.sum(dist < distThreshold**2):
            nodes.append(nodesTmp[i,:])
    nodes= np.array(nodes)
    #plt.figure(num="nodes")
    #plt.scatter(nodes[:,0], nodes[:,1])
    #plt.show()
    
    links= []
    layerLines.ResetReading()
    for feature in layerLines:
        line= feature.GetGeometryRef()
        num_points = line.GetPointCount()
        for i in range(num_points-1):
            x0, y0, z0= line.GetPoint(i)
            x1, y1, z1= line.GetPoint(i+1)
            dist0= (nodes[:,0]-x0)**2 + (nodes[:,1]-y0)**2
            dist1= (nodes[:,0]-x1)**2 + (nodes[:,1]-y1)**2
            node0= np.argmin(dist0)
            node1= np.argmin(dist1)
            links.append([node0,node1])
    links= np.array(links)
    print(links.shape)  
    
    a= np.arange(len(nodes))
    for i in a:
        print(i)
        # nodes.append([g])
        # print(i)
    # print(nodes.shape, a.shape)
    nodes2 = np.zeros((nodes.shape[0], 3))
    nodes2[:,1:] = nodes 
    nodes2[:,0] = np.arange(len(nodes))
    nodes3=nodes2 
    
    typenode= np.zeros((len(nodes), 1))
    nodes4= np.concatenate((nodes3, typenode), axis=1)
        
    reward= np.ones((len(nodes), 1))
    nodes5= np.concatenate((nodes4, reward), axis=1)
        
    nodes5[23][3]= 1
    
    b= np.arange(len(links))
    for i in b:
        print(i)
        # nodes.append([g])
        # print(i)
    # print(links.shape, b.shape)
    links2= np.zeros((links.shape[0], 3))
    links2[:,1:] = links 
    links2[:,0] = np.arange(len(links))
    links3=links2
    
    longitud= np.ones((len(links), 1))
    links3= np.concatenate((links3, longitud), axis=1)
        
    ancho= np.full((len(links), 1), 3)
    links4= np.concatenate((links3, ancho), axis=1)
    
    # print(nodes2.shape)
        
    np.savetxt('nodes5.csv', nodes5, delimiter=',', fmt='%d')
    np.savetxt('links4.csv', links4, delimiter=',', fmt='%d')
        
    plt.figure(num="links")
    for i in range(links.shape[0]):
        plt.plot([nodes[links[i,0],0],nodes[links[i,1],0]], [nodes[links[i,0],1],nodes[links[i,1],1]])
    plt.scatter(nodes[:,0], nodes[:,1])
    for i in range(nodes2.shape[0]):
        plt.text(nodes2[i, 1], nodes2[i, 2], "%d" % nodes2[i, 0])
    plt.show()
    
    # Unir dos puntos:
    for i in range(links.shape[0]):
        plt.plot([nodes[links[i,0],0],nodes[links[i,1],0]], [nodes[links[i,0],1],nodes[links[i,1],1]])
    plt.scatter(nodes[:,0], nodes[:,1])
    
    
    return nodes2, links2


def main():      
    inputShp= 'DataFiltrada_v4_utm.shp'      
    nodes3, links= GetNodesFromSHP(inputShp) 
    # print(nodes.shape, links.shape)
              
    return

if __name__ == "__main__":
    main()