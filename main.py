import cProfile
import numpy as np
import os
import time
import sys
sys.path.append('..')

import warnings
warnings.resetwarnings()
warnings.simplefilter('default') 

from SARSA import *     

project_directory = os.path.dirname(os.path.abspath(__file__))
outside_project_directory = os.path.join(project_directory, '..',)
sys.path.append(outside_project_directory)

def run(numSim0=0, numBlocks=1, simPerBlock=20, simulTime=10*60, foldername='Input'):
    """
    Runs the SARSA test simulation.

    Args:
        numSim0 (int): The starting simulation number.
        numBlocks (int): The number of simulation blocks.
        simPerBlock (int): The number of simulations per block.
        simulTime (int): The simulation time in seconds.

    Returns:
        None
    """
    t0 = time.time()
    
    agentsProfileName = os.path.join(foldername, "population.csv")
    nodesdbFile = os.path.join(foldername, "nodes.csv")
    linksdbFile = os.path.join(foldername, "edges.csv")
    transLinkdbFile = os.path.join(foldername, "actionsdb.csv") 
    transNodedbFile = os.path.join(foldername, "transitionsdb.csv")
    folderStateNames = os.path.join(foldername, "StatesMatrices")
    meanRayleighTest = 5 * 60 
    
    survivorsPerSim = []  
    
    if numSim0 == 0:
        randomChoiceRate = 0.99 
        optimalChoiceRate = 1.0 - randomChoiceRate
        sarsaTest = SARSA(agentsProfileName=agentsProfileName, 
                          nodesdbFile=nodesdbFile,
                          linksdbFile=linksdbFile, 
                          transLinkdbFile=transLinkdbFile, 
                          transNodedbFile=transNodedbFile,
                          meanRayleigh=meanRayleighTest,
                          ifCreateEmptyStateMatrix=True)        
        
        for t in range(int(min(sarsaTest.pedDB[:,9])), int(simulTime)):
            sarsaTest.initEvacuationAtTime()   
            sarsaTest.stepForward()
            optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate, optimalChoiceRate]))
            sarsaTest.checkTarget(ifOptChoice=optimalChoice)   
            if not t % 10:
                sarsaTest.computePedHistDenVelAtLinks()
                sarsaTest.updateVelocityAllPedestrians()
        
        stateMat, weights = sarsaTest.getStateMatricesAndWeights()
        
        print("\n\n ***** Simu %d (t= %.2f)*****" % (numSim0, (time.time()-t0)/60.))
        print("epsilon greedy - exploration: %f" % randomChoiceRate)
        print("survived pedestrians: %d" % np.sum(sarsaTest.pedDB[:,10] == 1))
        survivorsPerSim.append([numSim0, np.sum(sarsaTest.pedDB[:,10] == 1)])
        sarsaTest = None 
    
    print("Checkpoint")    
    
    numSim = numSim0 + 1  
    gleeFactor = 4.0 / simPerBlock 
    for b in range(numBlocks):
        for s in range(simPerBlock):
            randomChoiceRate = 1.0 / (gleeFactor * s + 1.0)
            optimalChoiceRate = 1.0 - randomChoiceRate
            sarsaTest = SARSA(agentsProfileName=agentsProfileName, 
                              nodesdbFile=nodesdbFile,
                              linksdbFile=linksdbFile, 
                              transLinkdbFile=transLinkdbFile, 
                              transNodedbFile=transNodedbFile,
                              meanRayleigh=meanRayleighTest,
                              ifCreateEmptyStateMatrix=False)
            
            sarsaTest.loadStateMatricesAndWeights(stateMat, weights)
            for t in range(int(min(sarsaTest.pedDB[:,9])), int(simulTime)):
                sarsaTest.initEvacuationAtTime()
                sarsaTest.stepForward()
                optimalChoice = bool(np.random.choice(2, p=[randomChoiceRate, optimalChoiceRate]))
                sarsaTest.checkTarget(ifOptChoice=optimalChoice)
                if not t % 10: 
                    sarsaTest.computePedHistDenVelAtLinks()
                    sarsaTest.updateVelocityAllPedestrians()
            
            stateMat, weights = sarsaTest.getStateMatricesAndWeights()
            
            print("\n\n ***** Simu %d (t= %.2f)*****" % (numSim, (time.time()-t0)/60.))
            print("epsilon greedy - exploration: %f" % randomChoiceRate)
            print("survived pedestrians: %d" % np.sum(sarsaTest.pedDB[:,10] == 1))
            survivorsPerSim.append([numSim, np.sum(sarsaTest.pedDB[:,10] == 1)])
            sarsaTest = None
            numSim += 1  
    
    outSurvivors = os.path.join(folderStateNames, "survivorsPerSim.csv")
    np.savetxt(outSurvivors, np.array(survivorsPerSim), delimiter=",", fmt="%d")  
    return


if __name__ == "__main__":  
    profiler = cProfile.Profile()
    profiler.enable()
    run()   
    profiler.disable()
    profiler.print_stats(sort='time')
     
        
        
        
        
