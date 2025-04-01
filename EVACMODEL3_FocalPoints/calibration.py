import numpy as np
import pandas as pd 
import os
import time
import sys
sys.path.append('..')
#project_directory = os.path.dirname(os.path.abspath(__file__))
#outside_project_directory = os.path.join(project_directory, '..',)
#sys.path.append(outside_project_directory)

from SARSA2024 import *    

def run(foldername, numSim0= 0, maxnumSim=0, numBlocks= 5, simPerBlock= 1000, simulTime = 60*60, popfile=1, meanrayleigh=5, reinforce_best_policy=True):
    t0 = time.time()
    
    agentsProfileName= os.path.join(foldername, f"population_{popfile}.csv")
    nodesdbFile=       os.path.join(foldername, "nodes.csv")
    linksdbFile=       os.path.join(foldername, "edges.csv")
    transLinkdbFile=   os.path.join(foldername, "actionsdb.csv")
    transNodedbFile=   os.path.join(foldername, "transitionsdb.csv")
    folderStateNames=  os.path.join(foldername, "StatesMatrices")
    meanRayleighTest = meanrayleigh*60
    
    survivorsPerSim= []    
    
    if numSim0 == 0:

        randomChoiceRate = 0.99
        optimalChoiceRate = 1.0 - randomChoiceRate
        sarsaTest = SARSA(agentsProfileName = agentsProfileName , 
                      nodesdbFile= nodesdbFile,
                      linksdbFile= linksdbFile, 
                      transLinkdbFile= transLinkdbFile, 
                      transNodedbFile= transNodedbFile,
                      meanRayleigh = meanRayleighTest)
        
        if maxnumSim != 0:
            namefile = os.path.join(folderStateNames , "sim_%09d.csv" % maxnumSim)
            new_namefile = os.path.join(folderStateNames , f"bestsim_{maxnumSim:09d}_{popfile}.csv")
            sarsaTest.loadStateMatrixFromFile(namefile = namefile)
            os.rename(namefile, new_namefile)
            os.system(f'rm -r ./{folderStateNames}/sim_*.csv')
        
        # sarsaTest.plotNetwork(labels= False, nodes=[])
        
        print(f'There is a total of {sarsaTest.numPedestrian} agents.')
        for t in range( int(min(sarsaTest.pedDB[:,9])) , int(simulTime)):
            # print("time",t)
            sarsaTest.initEvacuationAtTime()
            sarsaTest.stepForward()
            optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
            sarsaTest.checkTarget(ifOptChoice = optimalChoice)   
            if not t % 10:
                #print(t)
                # sarsaTest.updateVelocityAllPedestrians_and_Densities()
                sarsaTest.computePedHistDenVelAtLinks()
                sarsaTest.updateVelocityAllPedestrians()
        # sarsaTest.tdControl_EndEpisode()
        outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim0)
        sarsaTest.exportStateMatrix(outnamefile = outfile)
        print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim0, (time.time()-t0)/60. ))
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print("survived pedestrians: %d" % np.sum(sarsaTest.pedDB[:,10] == 1) )
        survivorsPerSim.append([numSim0, np.sum(sarsaTest.pedDB[:,10] == 1)])
        sarsaTest= None
    
    numSim= numSim0 +1
    gleeFactor= 1. / simPerBlock #ends calibration with 50/50 exploration/explotation
    for b in range(numBlocks):
        # gleeFactor= 1*(b+1) / simPerBlock
        for s in range(simPerBlock):
            outfile = os.path.join(folderStateNames , "sim_%09d.csv" % numSim)
            if os.path.exists(outfile):
                print("%s exist" % outfile)
                numSim += 1
                continue

            randomChoiceRate = 1.0/(gleeFactor*s + 1.0)
            optimalChoiceRate = 1.0 - randomChoiceRate
            sarsaTest = SARSA(agentsProfileName = agentsProfileName , 
                          nodesdbFile= nodesdbFile,
                          linksdbFile= linksdbFile, 
                          transLinkdbFile= transLinkdbFile, 
                          transNodedbFile= transNodedbFile,
                          meanRayleigh = meanRayleighTest)
            if reinforce_best_policy:
                namefile = os.path.join(folderStateNames , "sim_%09d.csv" % maxnumSim)
            else:
                namefile = os.path.join(folderStateNames , "sim_%09d.csv" % (numSim-1) )
            sarsaTest.loadStateMatrixFromFile(namefile = namefile)
            for t in range( int(min(sarsaTest.pedDB[:,9])) , int(simulTime)):
                sarsaTest.initEvacuationAtTime()
                sarsaTest.stepForward()
                optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate , optimalChoiceRate]))
                sarsaTest.checkTarget(ifOptChoice = optimalChoice)
                if not t % 10:
                    #print(t)
                    # sarsaTest.updateVelocityAllPedestrians_and_Densities()
                    sarsaTest.computePedHistDenVelAtLinks()
                    sarsaTest.updateVelocityAllPedestrians()
            # sarsaTest.tdControl_EndEpisode()
            
            sarsaTest.exportStateMatrix(outnamefile = outfile)
            print("\n\n ***** Simu %d (t= %.2f)*****" % ( numSim , (time.time()-t0)/60. ))
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print("survived pedestrians: %d" % np.sum(sarsaTest.pedDB[:,10] == 1) )
            survivorsPerSim.append([numSim, np.sum(sarsaTest.pedDB[:,10] == 1)])
            dff = pd.DataFrame(survivorsPerSim, columns=['numSim','survivors'])
            id = dff['survivors'].idxmax()
            maxnumSim = dff['numSim'][id]
            sarsaTest= None
            numSim += 1
    outSurvivors= os.path.join(folderStateNames, "survivorsPerSim.csv")
    np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter= ",", fmt= "%d" ) 
    return maxnumSim


if __name__ == "__main__":
    foldername = 'Input'
    numSim= 1
    numBlocks= 1  
    simPerBlock= 15000
    simulTime= 30*60
    run(foldername=foldername, numSim0= numSim, numBlocks= numBlocks, simPerBlock= simPerBlock, simulTime= simulTime)  
     
        
        
        
        
