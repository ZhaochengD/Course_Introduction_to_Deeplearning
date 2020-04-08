import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    batch_size = y_probs.shape[2]
    seq_length = y_probs.shape[1]
    
    best = np.argmax(y_probs, axis = 0)
    batch_best = []
    batch_prob = []
    
    batchIdx = 0
    for b in range(batch_size):
        batch = best[:, b]
        out = ''
        prob = 1
        for t in range(1, seq_length):
            prob *= y_probs[batch[t-1], t-1, batchIdx]
            if batch[t] != batch[t-1]:
                out += SymbolSets[batch[t-1] - 1] if batch[t-1] != 0 else ''
        out += SymbolSets[batch[-1] - 1] if batch[-1] != 0 else ''
        prob *= y_probs[batch[-1], -1, batchIdx]
        
        batch_best.append(out)
        batch_prob.append(prob)
        batchIdx += 1

    return batch_best[0], batch_prob[0]


##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
PathScore = {}
BlankPathScore = {}

def InitializePaths(SymbolSet, y):
    InitialBlankPathScore = {}
    InitialPathScore = {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ''
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1 
    InitialPathsWithFinalBlank = {path}
    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalSymbol = []
    for c in range(len(SymbolSet)): # This is the entire symbol set, without the blank
        path = SymbolSet[c]
        InitialPathScore[path] = y[c+1] # Score of symbol c at t=1 
        InitialPathsWithFinalSymbol.append(path) # Set addition
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
    global BlankPathScore
    global PathScore
    UpdatedPathsWithTerminalBlank = []
    UpdatedBlankPathScore = {}
    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence 
        UpdatedPathsWithTerminalBlank.append(path) # Set addition 
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank 
        # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.append(path) # Set addition
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
    global BlankPathScore
    global PathScore
    UpdatedPathsWithTerminalSymbol = []
    UpdatedPathScore = {}
    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        for c in range(len(SymbolSet)): # SymbolSet does not include blanks
            newpath = path + SymbolSet[c] # Concatenation 
            UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition 

            UpdatedPathScore[newpath] = BlankPathScore[path] * y[c + 1]

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
        for c in range(len(SymbolSet)): # SymbolSet does not include blanks
            newpath = path if (SymbolSet[c] == path[-1]) else path + SymbolSet[c] # Horizontal transitions don’t extend the sequence 
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[c + 1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition 
                UpdatedPathScore[newpath] = PathScore[path] * y[c + 1]
    
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    # First gather all the relevant scores
    scorelist = []
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
        
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])
    
    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist.sort(reverse = True) # In decreasing order
    cutoff = scorelist[BeamWidth - 1] if BeamWidth < len(scorelist) else scorelist[-1]

    PrunedPathsWithTerminalBlank = [] 
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
            PrunedPathsWithTerminalBlank.append(p) # Set addition 
            PrunedBlankPathScore[p] = BlankPathScore[p]

    PrunedPathsWithTerminalSymbol = []
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol.append(p) # Set addition 
            PrunedPathScore[p] = PathScore[p]
      
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    # All paths with terminal symbols will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    # Paths with terminal blanks will contribute scores to existing identical paths from 
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
            #FinalPathScore[p] += PathScore[p]
        else:
            MergedPaths.append(p) # Set addition
            FinalPathScore[p] = BlankPathScore[p]
            #FinalPathScore[p] = PathScore[p]

    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    global BlankPathScore
    global PathScore
    # Follow the pseudocode from lecture to complete beam search :-)

    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0].squeeze())

    
    # Subsequent time steps
    for t in range(1, y_probs.shape[1]):
    # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, 
                                                                                         NewBlankPathScore, NewPathScore, BeamWidth)

    # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,
                                                                  PathsWithTerminalSymbol, y_probs[:,t].squeeze())

    # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,t].squeeze())

    
    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank,NewBlankPathScore ,NewPathsWithTerminalSymbol, NewPathScore)

    # Pick best path
    BestPath = max(FinalPathScore, key=FinalPathScore.get)
    # return (bestPath, mergedPathScores)
    return BestPath, FinalPathScore



