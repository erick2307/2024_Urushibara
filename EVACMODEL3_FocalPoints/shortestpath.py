import sys
sys.path.append('..')  
import pandas as pd
import numpy as np
import os
import time

from SARSA2024 import *   

def run(foldername, timeSimulation= 30*60, popfile=1, meanrayleigh=5, video=False):   
    t0 = time.time() 
    meanRayleighTest = meanrayleigh*60
    agentsProfileName= os.path.join(foldername, f"population_{popfile}.csv") 
    nodesdbFile=       os.path.join(foldername, "nodes.csv")  
    linksdbFile=       os.path.join(foldername, "edges.csv")
    transLinkdbFile=   os.path.join(foldername, "actionsdb.csv")
    transNodedbFile=   os.path.join(foldername, "transitionsdb.csv") 
    shortesPathFile= os.path.join(foldername, "nextnode.csv")
    nameFileEvacuatedShortestPath= os.path.join(foldername, f"shortestPath_TimeVSEvacuated_{popfile}.csv")
    if video:
        os.makedirs(f'./{foldername}/Simulations', exist_ok=True)
        videoNamefile= os.path.join(foldername, "Simulations", f"shortestPath_{popfile}.avi")
    
    
    simSP = SARSA(agentsProfileName = agentsProfileName , 
                nodesdbFile= nodesdbFile,
                linksdbFile= linksdbFile, 
                transLinkdbFile= transLinkdbFile, 
                transNodedbFile= transNodedbFile,
                meanRayleigh = meanRayleighTest)
    
    simSP.loadShortestPathDB(shortesPathFile) 
    if video:
        simSP.setFigureCanvas()
    survivedAgents= []
    for t in range( int(min(simSP.pedDB[:,9])) , timeSimulation  ):       
        simSP.initEvacuationAtTime()
        simSP.stepForward()
        simSP.checkTargetShortestPath()
        if not t % 10:
            if video:
                simSP.getSnapshotV2(foldername = foldername)
            simSP.computePedHistDenVelAtLinks()
            simSP.updateVelocityAllPedestrians()
        survivedAgents.append([t, simSP.getNumberEvacuatedPed()])
            
    np.savetxt(nameFileEvacuatedShortestPath, np.array(survivedAgents) , delimiter= ",")
    print("survived pedestrians: %d" % np.sum(simSP.pedDB[:,10] == 1) )
    #convert survivedAgents to DataFrame with columns 'time', 'safe', 'time' column is in seconds
    survivedAgents_df = pd.DataFrame(survivedAgents, columns=['time', 'safe'])
    #remove rows where 'safe' are duplicated and keep the first instance
    survivedAgents_df_drop = survivedAgents_df.drop_duplicates(subset=['safe'], keep='first')
    evactime = survivedAgents_df_drop['time'].iloc[-1]
    #convert time which is an np.int64 reprsenting seconds to a string in the format HH:MM:SS
    print(f"Time of evacuation: {pd.to_datetime(evactime, unit='s').strftime('%H:%M:%S')}")
    if video:
        simSP.makeVideo(foldername = foldername, nameVideo = videoNamefile)
        simSP.destroyCanvas()
        simSP.deleteFigures(foldername = foldername)  
    return survivedAgents_df, evactime
 
if __name__ == "__main__":  
    run(foldername = 'Input', timeSimulation= 30*60)             
     
    