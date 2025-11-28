Issues 

- Feeder proportions don't sum to 1 for some reason 
    - If we were to fix this, we would have no overloaded feeders (based on overload amount I determined)
- Not sure what the grid capacity constraint should be - just made something to continue with optimization 
- Optimization sensitive to that amount of availabile grid capacity
- not all nodes and feeders represented. Feeder proportion matrix has feeders not in feeder capacity matrix
- Some nodes have no feeders based on feeder capacity matrix
- the totalled demand on feeders do not exceed capacity anywhere (this could definitely just be my fault because of how I determined the grid capacity constraint)
- need to watch the types of keys in each mapping element
    - Edited grid stress analysis to get load data with the feeder ids as strings rather than ints.
- Right now:
    - feeder node proportion is a super set of the feeder capacity 
    - node demand is a 
    - feeder node proportion is also a superset of the feeder demands
    - Major issue, feeder proportionality seems to zero out most of our nodes

- I may be screwing up the types of feeder_ids and dropping things incorrectly. 