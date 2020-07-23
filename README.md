# CSForest

Implementation of the cost-sensitive decision forest algorithm CSForest, which was published in:

*Siers, M. J., & Islam, M. Z. (2015). Software defect prediction using a cost sensitive decision forest and voting, and a potential solution to the class imbalance problem. Information Systems, 51, 62-71.*

This cost-sensitive decision forest was originally designed for software defect prediction datasets with two class values ("defective" and "not-defective"). It has been extended here to work with an arbitrary number of class values. The structure of the program is taken from our previous implementation of SysFor, upon which CSForest was based (and which in turn had been based on the Weka implementation of MetaCost). As this uses code from the SysFor implementation in Weka, it is worth noting that the algorithm does not search for "good" attributes beyond the second level of the tree.

For classification, the CSForest uses CSVoting, which is specified in the original paper.

## Installation

Either download CSForest from the Weka package manager, or download the latest release from the "Releases" section on the sidebar of Github.

## Compilation / Development

Set up a project in your IDE of choice, including weka.jar as a compile-time library.

## Valid options are:

`-L <minimum records in leaf>`
 Set minimum number of records for a leaf.
 (default 10)

`-N <no. trees>`
 Set number of trees to build.
 (default 60)

`-G <goodness threshold>`
 Set goodness threshold for attribute selection.
 (default 0.3)

`-S <separation threshold>`
 Set separation threshold for split point selection.
 (default 0.3)

`-C <confidence factor>`
 Set confidence for pruning.
 (default 0.25)

`-A`
Whether to calculate the total classification cost of the training dataset.
(default false)

`-cost-matrix <matrix>`
 The cost matrix in Matlab single line format.
 Expanded cost matrix takes form:
 ```
                   Actual class
                        | |
                        v v
                        a b
   class to be   =>  a  0 5
 classified as   =>  b  1 1
```
