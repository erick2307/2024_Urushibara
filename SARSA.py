import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
import multiprocessing as mp 
import itertools
plt.ioff() 

class SARSA:
    """
    SARSA class for implementing the SARSA algorithm for agent evacuation simulation.

    Parameters:
    - agentsProfileName (str): Name of the agents profile file.
    - nodesdbFile (str): Name of the nodes database file.
    - linksdbFile (str): Name of the links database file.
    - transLinkdbFile (str): Name of the transition links database file.
    - transNodedbFile (str): Name of the transition nodes database file.
    - meanRayleigh (float): Mean value for the Rayleigh distribution.
    - discount (float): Discount factor for future rewards.
    - surviveReward (float): Reward for surviving.
    - deadReward (float): Reward for dying.
    - stepReward (float): Reward for each time step.
    - errorLoc (float): Acceptable error between node and pedestrian coordinates.
    - ifCreateEmptyStateMatrix (bool): Flag to create empty state matrices for all nodes.

    Attributes:
    - surviveReward (float): Reward for surviving.
    - deadReward (float): Reward for dying.
    - stepReward (float): Reward for each time step.
    - discount (float): Discount factor for future rewards.
    - nodesdb (ndarray): Database of nodes.
    - linksdb (ndarray): Database of links.
    - transLinkdb (ndarray): Database of transition links.
    - transNodedb (ndarray): Database of transition nodes.
    - unitVectLinkdb (ndarray): Unit vector links.
    - populationAtLinks (ndarray): Population at each link.
    - popAtLink_HistParam (ndarray): Parameters for constructing histograms of populations at each link.
    - popHistPerLink (ndarray): Population histogram at each link.
    - denArrPerLink (ndarray): Density array at each link.
    - denLvlArrPerLink (ndarray): Density level array at each link.
    - maxDenLvlPerLink (ndarray): Maximum density level at each link.
    - speArrPerLink (ndarray): Speed array at each link.
    - evacuationNodes (ndarray): Evacuation nodes.
    - pedProfiles (ndarray): Pedestrian profiles.
    - numPedestrian (int): Number of pedestrians.
    - errorLoc (float): Acceptable error between node and pedestrian coordinates.
    - snapshotNumber (int): Snapshot number.
    - expeStat (list): Experienced states by pedestrian agents.
    - pedDB (ndarray): Pedestrian database.
    - time (float): Current time.
    - shortestPathDB (None or ndarray): Database with the shortest path.
    - stateMatPerNode (list): State matrices for all nodes.
    - weightLocation (dict): Weight for each location.

    Methods:
    - setStateMatricesPerNode(): Set state matrices for all nodes.
    - setTime(t): Set the time-step of the simulation.
    - computePedHistDenVelAtLinks(): Compute the histogram, density, and velocity at every link.
    - getPedHistAtLink(codeLink): Get the population histogram at a specific link.
    - getDenArrAtLink(codeLink): Get the density array at a specific link.
    - getVelArrAtLink(codeLink): Get the speed array at a specific link.
    - getStateIndexAtNode(codeNode): Get the state index at a specific node.
    - exportStateMatrix(outnamefile): Export the state matrix to a file.
    """

    def __init__(self, agentsProfileName="population.csv", 
                 nodesdbFile= "nodes.csv", 
                 linksdbFile= "edges.csv", 
                 transLinkdbFile= "actionsdb.csv", 
                 transNodedbFile= "transitionsdb.csv",
                 meanRayleigh=7*60, 
                 discount = 0.9,
                 surviveReward= 1000000, 
                 deadReward = -1000, 
                 stepReward = -1,
                 errorLoc = 2.0,
                 ifCreateEmptyStateMatrix= True):
        self.surviveReward = surviveReward
        self.deadReward = deadReward
        self.stepReward = stepReward
        self.discount = discount
        # node structure structure: [number, coordX, coordY, evacuationNode, rewardNode]
        self.nodesdb = np.loadtxt(nodesdbFile, delimiter=',', dtype= np.float64)  
        # linksdb format: [number, node1, node2, length, width]
        self.linksdb = np.loadtxt(linksdbFile, delimiter=',', dtype= np.float64) 
        # database of actions [currentNode, numberOfNodesTarget, linkConnectingNode1, linkConnectingNode2,...]
        self.transLinkdb = np.loadtxt(transLinkdbFile, delimiter=',', dtype=np.int64) 
        # database with possible transitions between nodes [currentNode, numberOfNodesTarget, nodeTarget1, nodeTarget2,...]
        self.transNodedb = np.loadtxt(transNodedbFile, delimiter=',', dtype=np.int64) 
        
        # unit vector links
        self.unitVectLinkdb= (self.nodesdb[ self.linksdb[:,2].astype(np.int64) , 1:3] - self.nodesdb[ self.linksdb[:,1].astype(np.int64) , 1:3])/np.linalg.norm( self.nodesdb[ self.linksdb[:,2].astype(np.int64) , 1:3] - self.nodesdb[ self.linksdb[:,1].astype(np.int64) , 1:3] ,axis=1).reshape(-1,1)
        self.unitVectLinkdb= np.nan_to_num(self.unitVectLinkdb) 
        
        # populationAtLinks: [linkNumber, numberOfAgentsAtLink]
        self.populationAtLinks = np.zeros((self.linksdb.shape[0], 2), dtype=np.int64) 
        self.populationAtLinks[:,0] = self.linksdb[:,0]
        
        # Parameters to construct histograms of polations at every link: 
        # (1) unit length, (2) number of units
        self.popAtLink_HistParam = np.zeros((self.linksdb.shape[0], 2), dtype=np.float64)
        self.popAtLink_HistParam[:,1] = np.round( self.linksdb[:,3] / 2. ) # assuming length units of about 2 meters
        indxZeroSegments= np.where( self.popAtLink_HistParam[:,1] == 0 )
        self.popAtLink_HistParam[indxZeroSegments,1]= 1
        self.popAtLink_HistParam[:,0] = self.linksdb[:,3] / self.popAtLink_HistParam[:,1]  
        
        # Separating memory for histogram information
        # the number of columns contains the larges number of segments of all the links
        self.popHistPerLink = np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+2 ), dtype=np.int64)
        self.denArrPerLink= np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+2 ) , dtype=np.float64)
        self.denLvlArrPerLink= np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+2 ), dtype="uint8")
        self.maxDenLvlPerLink= np.zeros(self.linksdb.shape[0] , dtype="uint8")
        self.speArrPerLink= np.zeros(( self.linksdb.shape[0] , int(max(self.popAtLink_HistParam[:,1]))+2 ), dtype=np.float64)
        
        # identifying evacuation nodes
        self.evacuationNodes = self.nodesdb[self.nodesdb[:,3] == 1,0].astype(np.int64)
        self.pedProfiles = np.loadtxt(agentsProfileName, delimiter=',', dtype=np.int64) # agents profile [age, gender, householdType, householdId, closestNodeNumber]
        self.numPedestrian = self.pedProfiles.shape[0]
        self.errorLoc = errorLoc   # acceptable error between coordinate of a node and a coordinate of a pedestrian
        self.snapshotNumber = 0
        
        # experienced  states by pedestrian agents
        # each component is a list containing [node, state_index, decision, time] 
        self.expeStat = [None] * self.numPedestrian
        
        # setting the pedestrian database
        # [(0)x0, (1)y0, (2)xtarget,(3)ytarget,(4)vx,(5)vy, (6)currentLink, (7)nodeTgt, (8)lastNode, (9)StartTimeEvacuation, (10)IfAlreadyEvacuated]
        self.pedDB = np.zeros((self.numPedestrian,11), dtype=np.float64) 
        # Assigning initial node
        self.pedDB[:,8] = self.pedProfiles[:,4]  
        # Before initiate evacuation, pedestrians do not have link
        # we assign a value of -2, which means, a pedestrian are not in a link
        self.pedDB[:,6] -= 2
        # Pedestrian already located in evacuation node:
        indxAlEv = np.isin(self.pedDB[:,8] , self.evacuationNodes)
        self.pedDB[ indxAlEv , 10] += 1
        # Assigning coordinates
        self.pedDB[:,0:2] = self.nodesdb[ self.pedProfiles[:,4] , 1:3 ] 
        # setting initial evacuation time for each pedestrian
        scaleRayleigh = meanRayleigh * (2/np.pi)**0.5
        self.pedDB[:,9] = np.round( np.random.rayleigh(scale = scaleRayleigh, size = self.pedDB.shape[0]) , decimals = 0 )
        self.time = min(self.pedDB[:,9])
        # setting database with the shortestpath, geometrically speaking, database
        self.shortestPathDB = None
        # setting state matrices for all nodes
        if ifCreateEmptyStateMatrix:
            self.setStateMatricesPerNode() 
    
    ########## Set environment ##########
    def setStateMatricesPerNode(self):
        """
        Set the state matrices for all nodes.
        """
        listNumLinks= np.unique( self.transLinkdb[:,1] )
        
        templates= {}
        self.weightLocation= {}
        for nl in listNumLinks:
            numR= 3**nl
            stateMat_tmp= np.zeros((numR,31), dtype=np.float64)
            stateMat_tmp[:,1:nl+1]= np.array(list( itertools.product ([0, 1,2], repeat=nl) ))
            stateMat_tmp[:,11:11+nl]= 0.5
            templates[nl]= stateMat_tmp 
            self.weightLocation[nl]= 3**np.arange(nl, dtype=np.int64)[::-1]

        self.stateMatPerNode= [None] * self.nodesdb.shape[0]
        
        for i in range( self.nodesdb.shape[0] ):
            nl= self.transLinkdb[i,1]
            self.stateMatPerNode[i]= np.copy( templates[nl] )  
            self.stateMatPerNode[i][:,0]= i
        return
    
    def setTime(self, t):
        """
        Set the time-step of the simulation.

        Parameters:
        - t (float): Time-step value.
        """
        self.time = t
        return
    
    def computePedHistDenVelAtLinks(self):
        """
        Compute the histogram, density, and velocity at every link.
        """
        
        occupLinksIndx = self.populationAtLinks[:,1] > 0
        unitL = self.popAtLink_HistParam[occupLinksIndx,0] 
        numComp= self.popAtLink_HistParam[occupLinksIndx,1].astype(np.int64) 
        lengthL= self.linksdb[occupLinksIndx, 3]
        width= self.linksdb[occupLinksIndx, 4]
        n0L= self.linksdb[occupLinksIndx, 1].astype(np.int64)
        x0L, y0L= self.nodesdb[n0L,1] , self.nodesdb[n0L,2]
        for i, ind in enumerate( np.where(occupLinksIndx)[0] ):
            dist= (( self.pedDB[ self.pedDB[:,6] == ind , 0]  - x0L[i])**2 + 
                   ( self.pedDB[ self.pedDB[:,6] == ind , 1] - y0L[i] )**2)**0.5
            dist = np.clip(dist, 0, lengthL[i])
            
            self.popHistPerLink[ind, :numComp[i] ], _= np.histogram(dist, bins= numComp[i], range= (0, lengthL[i]) ) 
            self.denArrPerLink[ind, :numComp[i]] = self.popHistPerLink[ind, :numComp[i] ].astype(np.float64) / (unitL[i] * width[i]) 
        self.speArrPerLink= np.clip( 1.388 - 0.396 * self.denArrPerLink , 0.2 , 1.19)
        self.denLvlArrPerLink[:,:]= 0
        self.denLvlArrPerLink[ self.denArrPerLink > 0.5 ] = 1
        self.denLvlArrPerLink[ self.denArrPerLink > 3.0 ] = 2
        
        emptyLinksIndx = self.populationAtLinks[:,1] == 0
        self.popHistPerLink[emptyLinksIndx] = 0
        self.denArrPerLink[emptyLinksIndx] = 0
        self.speArrPerLink[emptyLinksIndx] = 1.19
        self.denLvlArrPerLink[emptyLinksIndx] = 0
        self.maxDenLvlPerLink[emptyLinksIndx] = 0
        
        self.maxDenLvlPerLink= np.max(self.denLvlArrPerLink, axis=1)

        return
    
    def getPedHistAtLink(self, codeLink):
        """
        Get the population histogram at a specific link.

        Parameters:
        - codeLink (int): Code of the link.

        Returns:
        - ndarray: Population histogram at the specified link.
        """
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.popHistPerLink[codeLink, :numComp] 
    
    def getDenArrAtLink(self, codeLink):
        """
        Get the density array at a specific link.

        Parameters:
        - codeLink (int): Code of the link.

        Returns:
        - ndarray: Density array at the specified link.
        """
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.denArrPerLink[codeLink, :numComp]
    
    def getVelArrAtLink(self, codeLink):
        """
        Get the speed array at a specific link.

        Parameters:
        - codeLink (int): Code of the link.

        Returns:
        - ndarray: Speed array at the specified link.
        """
        numComp= int( self.popAtLink_HistParam[codeLink,1] )
        return self.speArrPerLink[codeLink, :numComp]
    
    def getStateIndexAtNode(self, codeNode):
        """
        Get the state index at a specific node.

        Parameters:
        - codeNode (int): Code of the node.

        Returns:
        - int: State index at the specified node.
        """
        linksdb = self.transLinkdb[codeNode,:]
        linkDensityAr= np.zeros(linksdb[1])
        for l in range(2,2+linksdb[1]):
            linkDensityAr[l-2]= self.maxDenLvlPerLink[linksdb[l]]
        return int( np.dot( linkDensityAr , self.weightLocation[ linksdb[1] ]) )
    
    
    def exportStateMatrix(self, outnamefile="stateMatrix.npz"):
        """
        Export the state matrix to a file.

        Parameters:
        - outnamefile (str): Name of the output file.
        """
        np.savez(outnamefile, *self.stateMatPerNode)
        return
    
    def loadStateMatrixFromFile(self, namefile):
        """
        Updates the matrix "stateMatPerNode" from a file. Useful to the learning process,
        in which we want the stateMat from previous simulations
        """
        loaded_states = np.load(namefile)
        self.stateMatPerNode= [loaded_states[f'arr_{i}'] for i in range(len(loaded_states.files))]
        return  
    
    def getStateMatricesAndWeights(self):
        return self.stateMatPerNode, self.weightLocation
    
    def loadStateMatricesAndWeights(self, stateMatPerNode, weightLocation):
        self.stateMatPerNode= stateMatPerNode
        self.weightLocation= weightLocation
        return
    
    def exportAgentDBatTimet(self,outnamefile):
        """
        Exports the matrix "pedDb" in the file "outnamefile" in csv format

        """       
        np.savetxt(outnamefile, self.pedDB, delimiter=',', fmt=["%.6f","%.6f","%.6f","%.6f","%.3f","%.3f","%d","%d","%d","%d","%d"])
        return

    def getPopulationAtLinks(self):
        """
        return the matrix "populationAtLinks", the number of pedestrian at links.
        """
        return self.populationAtLinks
    
    def resizePedestrianDB(self, size):
        indx= np.random.choice( self.pedDB.shape[0] , size= size )
        self.pedDB = self.pedDB[indx, :]
        return
    
    ######### Move a unit step in the simulation #########
    def stepForward(self, dt=1):
        """
        Moves the simulation one step forward. That is, 
        the pedestrian moves according to their velocity for a period of time "dt".
        it also updates the variable "time".
        """
        self.pedDB[ : , 0:2 ] += dt * self.pedDB[ : , 4:6 ]
        self.time += dt
        return
    
    def checkTarget(self, ifOptChoice = False):
        """
        Verifies if the pedestrians arrived to their node target.
        For those who already arrived, the function assign the next target.
        """
        # Computes the distance of the pedestrian coordinates to the node trget coordinates:
        error = ( (self.pedDB[ : , 0 ] - self.pedDB[ : , 2 ])**2 + (self.pedDB[ :  , 1 ] - self.pedDB[ : , 3 ])**2 )**0.5
        indx = np.where(error <= self.errorLoc)[0]
        for i in indx:
            # do not check target if the experience db is empty
            if not self.expeStat[i]:
                continue
            # check if pedestrian already evacuated
            if self.pedDB[i,10]:
                continue
            self.updateTarget(i, ifOptChoice = ifOptChoice)
        return
    
    def updateVelocityV2(self, pedIndx):
        codeLink= int( self.pedDB[pedIndx, 6] )
        
        if codeLink == -1:
            return
        else:
            n0L= int( self.linksdb[codeLink, 1] )
            x0L, y0L= self.nodesdb[n0L,1] , self.nodesdb[n0L,2]
            dist= ( (self.pedDB[pedIndx,0] - x0L)**2 + (self.pedDB[pedIndx,1] - y0L)**2 )**0.5
            unitL= self.popAtLink_HistParam[codeLink,0]
            
            try:
                xAtLink= int( np.floor( dist / unitL) ) 
            except:
                print(dist, unitL)
                print(self.popAtLink_HistParam[codeLink])
                print(self.linksdb[codeLink])
                print(codeLink)
            speed= self.speArrPerLink[codeLink, xAtLink] + np.random.rand()*0.02 - 0.01  
            unitDir = (self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2]) / np.linalg.norm(self.pedDB[pedIndx,2:4] - self.pedDB[pedIndx,:2])
            vel_arr = speed * unitDir
            
            self.pedDB[pedIndx, 4:6] = vel_arr
        return
    
    def updateVelocityAllPedestrians(self):
        # indxAtEvaPoi= self.pedDB[:,10] == 0
        # indxIniEva= self.time > self.pedDB[:,9] 
        # indxBoth = np.where( np.logical_and(indxAtEvaPoi, indxIniEva) )[0]
        
        # for ind in indxBoth:
        #     self.updateVelocityV2(ind)
        
        ## New work april 18 2024
        occupLinksIndx = self.populationAtLinks[:,1] > 0
        unitL = self.popAtLink_HistParam[occupLinksIndx,0] 
        lengthL= self.linksdb[occupLinksIndx, 3]
        n0L= self.linksdb[occupLinksIndx, 1].astype(np.int64)
        x0L, y0L= self.nodesdb[n0L,1] , self.nodesdb[n0L,2]
        for i, ind in enumerate( np.where(occupLinksIndx)[0] ):
            dist= np.clip( (( self.pedDB[ self.pedDB[:,6] == ind , 0]  - x0L[i])**2 + 
                           ( self.pedDB[ self.pedDB[:,6] == ind , 1] - y0L[i] )**2)**0.5 ,
                          0, lengthL[i])
            # print( dist , unitL[i] , dist / unitL[i]) 
            xAtLink= np.floor( dist / unitL[i]).astype(np.int64)
            # print(dist) #<= this is the problem
            speed= self.speArrPerLink[ind, xAtLink] + np.random.rand(xAtLink.shape[0])*0.02 - 0.01 
            unitDir = self.pedDB[ self.pedDB[:,6] == ind , 4:6] / np.linalg.norm(self.pedDB[ self.pedDB[:,6] == ind , 4:6], axis=1).reshape(-1,1)
            self.pedDB[ self.pedDB[:,6] == ind , 4:6] = unitDir * speed.reshape(-1,1)

        return

    def initEvacuationAtTime(self):
        """
        this function updates the velocity of the agents according to their initial evacuation time.
        That is, the function will identify the agents whose initial evacuation time equals the current parameter "self.time".
        Then, the velocity of these agents are updates (because they have velocity zero at first).
        furthermore, here the first experienced state is recorded.
        """
        # Find pedestrian initiating evacuation
        # indxPed = np.where(self.pedDB[:,9] == self.time)[0]
        # # Check if there are pdestrians starting evacuation
        # if len(indxPed) != 0:
        #     for i in indxPed:
        #         node0 = int(self.pedDB[i,8]) # current node
        #         indxTgt = np.random.choice( int(self.transNodedb[node0,1]) ) # random choise for the next node
        #         nodeTgt = self.transNodedb[node0, 2+indxTgt] # next number node
        #         link = self.transLinkdb[node0, 2+indxTgt] # next link code
        #         self.pedDB[i,7] = nodeTgt
        #         self.pedDB[i,6] = link
        #         self.pedDB[i, 2:4] = self.nodesdb[nodeTgt, 1:3] # coordinates of the next target (next node)
        #         # state [node, state_index, choice, time]
        #         firstState = np.zeros(4, dtype = np.int64)
        #         firstState[2] = int(indxTgt)
        #         self.updateVelocityV2(i)  #previous: self.updateVelocity(i, int(self.pedDB[i,6]))
        #         # Get state code at starting node
        #         indxStat = self.getStateIndexAtNode(int(self.pedDB[i,8]))
                
        #         firstState[0]= int(node0)
        #         firstState[1]= int(indxStat)
        #         firstState[3]= int(self.time)
        #         self.expeStat[i] = [firstState] 
        #         # Report a new pedestrian enter link, but first we check if the pedestrian arrived an evacuation-node
        #         if int(self.pedDB[i,6]) >= 0:
        #             self.populationAtLinks[int(self.pedDB[i,6]), 1] += 1
        
        indxPed = self.pedDB[:,9] == self.time
        if np.sum(indxPed):
            node0= self.pedDB[indxPed,8].astype(np.int64)
            # print(node0)
            indxTgt= np.array( [np.random.choice(n) for n in self.transNodedb[node0,1].astype(np.int64)] )
            # print(indxTgt)
            nodeTgt = self.transNodedb[node0, 2+indxTgt]
            link = self.transLinkdb[node0, 2+indxTgt]
            self.pedDB[indxPed,7] = nodeTgt
            self.pedDB[indxPed,6] = link
            self.pedDB[indxPed, 2:4] = self.nodesdb[nodeTgt, 1:3]
            unit_vector= (self.pedDB[indxPed, 2:4] - self.pedDB[indxPed, :2])/np.linalg.norm( self.pedDB[indxPed, 2:4] - self.pedDB[indxPed, :2] , axis=1).reshape(-1,1)
            self.pedDB[indxPed, 4:6]= 1.19 * unit_vector 
            # print(unit_vector)
            # print(self.pedDB[indxPed, 4:6])
            # print(link)
            # print(node0.shape, unit_vector.shape)
            firstStates= np.zeros( (node0.shape[0],4), dtype=np.int64)
            firstStates[:,0]= node0
            firstStates[:,2]= indxTgt
            firstStates[:,3]= int(self.time)
            # print(firstStates)
            # print( self.pedDB[indxPed, : ] )
            # print(" ")
            for i, ind in enumerate( np.where(indxPed)[0] ):
                # print(i, ind)
                indxStat = self.getStateIndexAtNode(node0[i]) 
                firstStates[i,1]= indxStat 
                self.expeStat[ind]= [ firstStates[i,:] ]
                if link[i] >= 0:
                    self.populationAtLinks[int(self.pedDB[ind,6]), 1] += 1
            # print(self.time)
            # print(self.pedDB[indxPed, :])
            # print(" ")
        return
    
    def updateTarget(self, pedIndx, ifOptChoice = False):
        """
        Updates the node target of pedestrian pedIndx.
        It considers whether the pedestrian uses optimal (exploting) 
        or random (exploring) approach. 
        """
        # Check if ped is in evacuation node:
        if self.pedDB[pedIndx,10]:
            return
        else:
            # New start node is now previous target node:
            node0 = int(self.pedDB[pedIndx , 7])
            x0_arr = np.array([ self.nodesdb[node0,1], self.nodesdb[node0,2] ])
            stateIndx = self.getStateIndexAtNode(node0)
            Qval_arr = self.stateMatPerNode[node0][stateIndx , 11:11+int(self.transLinkdb[node0,1])]
            
            # If optimal choice, it selects the action with the largest action-value:
            if ifOptChoice:
                indxTgt = np.where( Qval_arr == max(Qval_arr) )[0][0]
            else:
                indxTgt = np.random.choice(int(self.transNodedb[node0,1]))
            # Get chosen link and new target node:
            link = self.transLinkdb[node0, 2+indxTgt]
            nodeTgt = self.transNodedb[node0, 2+indxTgt]

            if nodeTgt == node0:
                xTgt_arr = x0_arr
                vel_arr = np.array([0,0])
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.pedDB[pedIndx,10] = 1
                expeStatAndVal = np.array([node0, stateIndx, indxTgt, self.time], dtype=np.int64)
                self.pedDB[pedIndx , :9] = np.array( [x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0] )
            else:
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.populationAtLinks[link, 1] += 1
                xTgt_arr = np.array([ self.nodesdb[nodeTgt,1], self.nodesdb[nodeTgt,2] ])
                
                vel_arr = np.array([0,0])
                expeStatAndVal = np.array([node0, stateIndx, indxTgt, self.time], dtype=np.int64)
                self.pedDB[pedIndx , :9] = np.array( [x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0] )
                self.updateVelocityV2(pedIndx)
            
            self.expeStat[pedIndx].append(expeStatAndVal) 
            self.tdControl(pedIndx)
        return
    
    def tdControl(self, pedIndx, alpha= 0.05):
        
        """
        This funciton represents the main change between SARSA and MonteCarlo.
        Here we update the variable "stateMatPerNode" during the episode, rather than at the end.

        """
        trackPed = np.array(self.expeStat[pedIndx], dtype="int")
        # current and previous states
        current_S= trackPed[-1,:2]
        pre_S= trackPed[-2,:2]
        # current and previous actions
        current_A= trackPed[-1,2]
        pre_A= trackPed[-2,2]
        # current and pre time
        current_t = trackPed[-1,3]
        pre_t = trackPed[-2,3]
        
        if self.pedDB[pedIndx,10]:
            currentReward= self.surviveReward
            self.stateMatPerNode[current_S[0]][current_S[1], 11 + current_A] += alpha * (currentReward - self.stateMatPerNode[current_S[0]][current_S[1], 11 + current_A])
            self.stateMatPerNode[current_S[0]][current_S[1], 21 + current_A] += 1
        
        preReward= self.stepReward * (current_t - pre_t)
        self.stateMatPerNode[pre_S[0]][pre_S[1], 11 + pre_A] += alpha * (preReward + 
                                                                         self.discount * self.stateMatPerNode[current_S[0]][current_S[1], 11 + current_A] - 
                                                                         self.stateMatPerNode[pre_S[0]][pre_S[1], 11 + pre_A])
        self.stateMatPerNode[pre_S[0]][pre_S[1], 21 + pre_A] += 1
        return 
    
    
    ########## UTILS ###########
    
    def getNumberEvacuatedPed(self):
        return np.sum(self.pedDB[:,10] == 1)
    
    ########## functions to use shortest path
    
    def loadShortestPathDB(self, namefile="closePath_nextnode.csv"):
        self.shortestPathDB = np.loadtxt(namefile, delimiter=",", dtype=np.int64)
        return
    
    def updateTargetShortestPath(self, pedIndx):  
        if self.pedDB[pedIndx, 10]:
            return
        else:
            node0 = int(self.pedDB[pedIndx , 7])
            x0_arr = np.array([ self.nodesdb[node0,1], self.nodesdb[node0,2] ])
            
            nodeTgt = self.shortestPathDB[node0,1]
            if nodeTgt == -9999:
                return
            numNodesLinked = self.transNodedb[node0, 1]
            nodesLinked = self.transNodedb[node0, 2:2+numNodesLinked]
            # print(nodesLinked)
            # print(nodeTgt)
            # print(np.where(nodesLinked == nodeTgt))
            indxTgt = np.where(nodesLinked == nodeTgt)[0][0]
            link = self.transLinkdb[node0, 2+indxTgt]
            
            if nodeTgt == node0:
                xTgt_arr = x0_arr
                vel_arr = np.array([0,0])
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.pedDB[pedIndx,10] = 1
                self.pedDB[pedIndx , :9] = np.array( [x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0] )
                
            else:
                self.populationAtLinks[int(self.pedDB[pedIndx,6]), 1] -= 1
                self.populationAtLinks[link, 1] += 1
                xTgt_arr = np.array([ self.nodesdb[nodeTgt,1], self.nodesdb[nodeTgt,2] ])                
                vel_arr = np.array([0,0])
                self.pedDB[pedIndx , :9] = np.array( [x0_arr[0], x0_arr[1], xTgt_arr[0], xTgt_arr[1], vel_arr[0], vel_arr[1], link, nodeTgt, node0] )                
                self.updateVelocityV2(pedIndx)  # HERE Is THE BUUG            
        return
    
    def checkTargetShortestPath(self):
        # print("Cheking target")
        
        error = ( (self.pedDB[ : , 0 ] - self.pedDB[ : , 2 ])**2 + (self.pedDB[ :  , 1 ] - self.pedDB[ : , 3 ])**2 )**0.5
        indx = np.where(error <= self.errorLoc)[0]
        # print("indx", indx)
        for i in indx:
            #copied from checkTarget 20204.10.26
            # do not check target if the experience db is empty
            # if not self.expeStat[i]:
            #     continue
            # check if pedestrian already evacuated
            if self.pedDB[i,10]:
                continue
            self.updateTargetShortestPath(i)
        return
    
            # if self.time < self.pedDB[i,9]:
            #     continue
            
            # if self.pedDB[i,10]:
            #     continue
            
            # self.updateTargetShortestPath(i)   # here is the bug!!!
            # print("2",self.pedDB[indxNE,4])
        return

 
    ########## Visualization functions ##########
    def setFigureCanvas(self):
        self.fig, self.ax = plt.subplots(figsize=(12,5))
        for i in range(self.linksdb.shape[0]):
            self.ax.plot([self.nodesdb[int(self.linksdb[i,1]),1], self.nodesdb[int(self.linksdb[i,2]),1]],[self.nodesdb[int(self.linksdb[i,1]),2], self.nodesdb[int(self.linksdb[i,2]),2]], c='k', lw=1)
        indxEv = self.nodesdb[:,3] == 1.0
        self.p1, = self.ax.plot(self.nodesdb[indxEv,1], self.nodesdb[indxEv,2], 'rD', ms=4, mfc= "none", mec= "r")
        indx = np.where(self.pedDB[:,9] <= self.time)[0]
        speed= ( (self.pedDB[indx,4])**2 + (self.pedDB[indx,5])**2)**0.5
        self.p2 = self.ax.scatter(self.pedDB[indx,0], self.pedDB[indx,1], 
                                  c= speed, s=10, vmin=0.1, vmax=1.3, 
                                  cmap="jet_r", edgecolors='none')
        self.fig.colorbar(self.p2) 
        self.ax.axis("equal")
        self.ax.set_axis_off()
        self.lblCoord = np.array([max(self.nodesdb[:,1]), max(self.nodesdb[:,2])])
        self.labelTime = self.fig.text( 0, 0, " ")
        return
    
    def getSnapshotV2(self, ifNodeAround= False, nodeAroundCode= 1070):
        self.p2.remove()
        indx = np.where(self.pedDB[:,9] <= self.time)[0]
        speed= ( (self.pedDB[indx,4])**2 + (self.pedDB[indx,5])**2)**0.5
        self.p2 = self.ax.scatter(self.pedDB[indx,0], self.pedDB[indx,1], 
                                  c= speed, s=8, vmin=0.1, vmax=1.3, 
                                  cmap="jet_r", edgecolors='none') 
        self.labelTime.remove()
        self.labelTime = self.fig.text( 0, 0, "t = %.2f min; evacuated: %d of %d" % (self.time/60., np.sum(self.pedDB[:,10] == 1), self.pedDB.shape[0]))
        if ifNodeAround:
            coord= self.nodesdb[nodeAroundCode,1:3]
            self.ax.set_xlim(coord[0]-200,coord[0]+200)
            self.ax.set_ylim(coord[1]-200,coord[1]+200)
        self.fig.savefig(os.path.join("Input","Figures", "Figure_%04d.png" % self.snapshotNumber), 
                         bbox_inches="tight", dpi=150)
        
        self.snapshotNumber += 1
        return
    
    
    def makeVideo(self, nameVideo = "Simul.avi"):
        listImagesUS = glob.glob( os.path.join("Input", "Figures", "*png"))
        numSS_ar= np.zeros( len(listImagesUS) , dtype= "int")
        for i, li in enumerate(listImagesUS):
            numSS_ar[i]= int( li[-8:-4] ) 
        indSort= np.argsort(numSS_ar)
        listImages= []
        for i in indSort:
            listImages.append( listImagesUS[i] )
            
        img = cv2.imread(listImages[0])
        height_im, width_im, layers_im = img.shape
        video = cv2.VideoWriter(nameVideo, cv2.VideoWriter_fourcc('M','J','P','G'),15,(width_im, height_im))  # Only works with openCV3
        for im in listImages:
            print(im)
            img = cv2.imread(im)
            video.write(img)
        cv2.destroyAllWindows()
        video.release()
        return
    
    def destroyCanvas(self):
        self.p1 = None
        self.p2 = None
        self.ax = None
        self.fig = None
        return
    
    def deleteFigures(self):
        figures = glob.glob( os.path.join("Input","Figures","*") ) 
        for f in figures:
            os.remove(f)
    
    def plotNetwork(self):
        colorNode = ["b","r"] 
        plt.figure(num="Network",figsize=(5,4))
        for i in range(self.linksdb.shape[0]):
            plt.plot([self.nodesdb[int(self.linksdb[i,1]),1], self.nodesdb[int(self.linksdb[i,2]),1]],[self.nodesdb[int(self.linksdb[i,1]),2], self.nodesdb[int(self.linksdb[i,2]),2]], c='k', lw=1)
            plt.text(0.5*(self.nodesdb[int(self.linksdb[i,1]),1] + self.nodesdb[int(self.linksdb[i,2]),1]) , 0.5*(self.nodesdb[int(self.linksdb[i,1]),2] + self.nodesdb[int(self.linksdb[i,2]),2]), 
                    "%d" % self.linksdb[i,0], fontsize=8, color='g')
        for i in range(self.nodesdb.shape[0]):
            plt.text(self.nodesdb[i,1], self.nodesdb[i,2], self.nodesdb[i,0], fontsize = 7, color=colorNode[ int(self.nodesdb[i,3]) ])
        indxZeroActions = np.where(self.transLinkdb[:,1] != 0)[0]
        plt.scatter(self.nodesdb[indxZeroActions][:,1], self.nodesdb[indxZeroActions][:,2], s=20, edgecolor='k', linewidths=0.0)
        
        indxEv = self.nodesdb[:,3] == 1.0
        plt.scatter(self.nodesdb[indxEv,1], self.nodesdb[indxEv,2], s=20, c="r")
        
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def plotBestChoices(self, ifUpdateLinkDensity= False, linkList=[], densityLvlList= [],
                        ifSaveFile= False, baseNameFile= "BC"):
        if ifUpdateLinkDensity:
            for i, linkId in enumerate(linkList):
                self.denLvlArrPerLink[linkId, :]= 0
                self.denLvlArrPerLink[linkId, 0]= densityLvlList[i]
        
        colorNode = ["b","g"] 
        colorLink = ["k","g","r"] 
        plt.figure(num= "NetworkChoice_time%04d" % self.time,figsize=(5,4)) 
        for i in range(self.linksdb.shape[0]):
            densityLink= int( np.max( self.denLvlArrPerLink[i, :] ) )
            plt.plot([self.nodesdb[int(self.linksdb[i,1]),1], self.nodesdb[int(self.linksdb[i,2]),1]],[self.nodesdb[int(self.linksdb[i,1]),2], self.nodesdb[int(self.linksdb[i,2]),2]], 
                     c= colorLink[densityLink], lw=1)
        for i in range(self.nodesdb.shape[0]):
            plt.text(self.nodesdb[i,1]+1.0, self.nodesdb[i,2]+1.0, int(self.nodesdb[i,0]), fontsize = 7, color=colorNode[ int(self.nodesdb[i,3]) ])
        indxZeroActions = np.where(self.transLinkdb[:,1] != 0)[0]
        plt.scatter(self.nodesdb[indxZeroActions][:,1], self.nodesdb[indxZeroActions][:,2], s=20, edgecolor='k', linewidths=0.0)
        
        indxEv = self.nodesdb[:,3] == 1.0
        plt.scatter(self.nodesdb[indxEv,1], self.nodesdb[indxEv,2], s=20, c="g")

        for i in range(self.nodesdb.shape[0]):
            if self.nodesdb[i,3]:
                continue
            stateNodeIndex= self.getStateIndexAtNode(i)
            stateNode= self.stateMatPerNode[i][stateNodeIndex,:]
            indxChoice= np.argmax(stateNode[11:21])

            nodeTarget= self.transNodedb[i, indxChoice+2]
            x0, y0= self.nodesdb[i,1], self.nodesdb[i,2]
            x1, y1= self.nodesdb[nodeTarget,1], self.nodesdb[nodeTarget,2]
            x1Sc= x0 + 0.33*(x1-x0)
            y1Sc= y0 + 0.33*(y1-y0)
            
            plt.arrow(x0,y0, 0.2*(x1-x0), 0.2*(y1-y0), color="k",
                      width= 1)
        
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        
        if ifSaveFile:
            fileName= os.path.join("Input","Figures_BestChoice",baseNameFile + "_time%04d.png" % self.time )
            plt.savefig(fileName, bbox_inches="tight", dpi= 150)
        
        plt.clf()
        plt.close("NetworkChoice_time%04d" % self.time)
        return

###############################################################################


if __name__ == "__main__":
    print("start")
