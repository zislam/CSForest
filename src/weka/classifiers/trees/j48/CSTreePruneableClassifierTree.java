/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    CSTreePruneableClassifierTree.java
 *
 */
package weka.classifiers.trees.j48;

import weka.classifiers.CostMatrix;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a tree structure that can be pruned using cost-sensitive
 * procedures.
 *
 * @author modified by Michael Furner (mfurner@csu.edu.au) (originally by Eibe
 * Frank (eibe@cs.waikato.ac.nz))
 */
public class CSTreePruneableClassifierTree
        extends CostBasedClassifierTree {

    /**
     * for serialization
     */
    static final long serialVersionUID = -4813820170260388194L;

    /**
     * True if the tree is to be pruned.
     */
    protected boolean m_pruneTheTree = false;

    /**
     * True if the tree is to be collapsed.
     */
    protected boolean m_collapseTheTree = false;

    /**
     * The confidence factor for pruning.
     */
    protected float m_CF = 0.25f;

    /**
     * Is subtree raising to be performed?
     */
    protected boolean m_subtreeRaising = true;

    /**
     * Cleanup after the tree has been built.
     */
    protected boolean m_cleanup = true;

    /**
     * Constructor for pruneable tree structure. Stores reference to associated
     * training data at each node.
     *
     * @param toSelectLocModel selection method for local splitting model
     * @param pruneTree true if the tree is to be pruned
     * @param cf the confidence factor for pruning
     * @param raiseTree
     * @param cleanup
     * @param collapseTree
     * @param cm the cost matrix
     * @throws Exception if something goes wrong
     */
    public CSTreePruneableClassifierTree(ModelSelection toSelectLocModel,
            boolean pruneTree, float cf,
            boolean raiseTree,
            boolean cleanup,
            boolean collapseTree, CostMatrix cm)
            throws Exception {

        super(toSelectLocModel, cm);

        m_pruneTheTree = pruneTree;
        m_CF = cf;
        m_subtreeRaising = raiseTree;
        m_cleanup = cleanup;
        m_collapseTheTree = collapseTree;
    }

    /**
     * Returns default capabilities of the classifier tree.
     *
     * @return the capabilities of this classifier tree
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Method for building a pruneable classifier tree.
     *
     * @param data the data for building the tree
     * @throws Exception if something goes wrong
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, m_subtreeRaising || !m_cleanup);
        if (m_pruneTheTree) {
            pruneViaCost();
        }
        if (m_cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    /**
     * Collapses a tree to a node if training error doesn't increase.
     */
    public final void collapse() {

        double errorsOfSubtree;
        double errorsOfTree;
        int i;

        if (!m_isLeaf) {
            errorsOfSubtree = getTrainingErrors();
            errorsOfTree = localModel().distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree - 1E-3) {

                // Free adjacent trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for tree.
                m_localModel = new NoSplit(localModel().distribution());
            } else {
                for (i = 0; i < m_sons.length; i++) {
                    son(i).collapse();
                }
            }
        }
    }
    
    /**
     * Prunes a tree using a cost-based pruning procedure.
     *
     * @throws Exception if something goes wrong
     */
    public void pruneViaCost() throws Exception {
        double costLargestBranch;
        double costLeaf;
        double costTree;
        int indexOfLargestBranch;
        CSTreePruneableClassifierTree largestBranch;
        int i;

        if (!m_isLeaf) {

            // Prune all subtrees.
            for (i = 0; i < m_sons.length; i++) {
                son(i).pruneViaCost();
            }

            // Compute error for largest branch
            indexOfLargestBranch = localModel().distribution().maxBag();
            if (m_subtreeRaising) {
                costLargestBranch = son(indexOfLargestBranch).
                        getCostForBranch((Instances) m_train);
            } else {
                costLargestBranch = Double.MAX_VALUE;
            }

            // Compute error if this Tree would be leaf
            costLeaf
                    = getCostForDistribution(localModel().distribution()); //parentErrorCost

            // Compute error for the whole subtree
            costTree = getCost(); //leafErrorCost

            // Decide if leaf is best choice.
            if (Utils.sm(costLeaf, costTree)
                    && Utils.smOrEq(costLeaf, costLargestBranch + 0.1)) {

                // Free son Trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for node.
                m_localModel = new NoSplit(localModel().distribution());
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(costLargestBranch, costTree + 0.1)) {
                largestBranch = son(indexOfLargestBranch);
                m_sons = largestBranch.m_sons;
                m_localModel = largestBranch.localModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                pruneViaCost();
            }
        }
    }


    /**
     * Returns a newly created tree.
     *
     * @param data the data to work with
     * @return the new tree
     * @throws Exception if something goes wrong
     */
    protected CostBasedClassifierTree getNewTree(Instances data) throws Exception {

        CSTreePruneableClassifierTree newTree
                = new CSTreePruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_CF,
                        m_subtreeRaising, m_cleanup, m_collapseTheTree, m_costMatrix);
        newTree.buildTree((Instances) data, m_subtreeRaising || !m_cleanup);

        return newTree;
    }
    
    private double getCost() throws Exception {

        double cost = 0;
        int i;

        if (m_isLeaf) {
            return getCostForDistribution(localModel().distribution());
        } else {
            for (i = 0; i < m_sons.length; i++) {
                cost = cost + son(i).getCost();
            }
            return cost;
        }
    }
    
    private double getCostForBranch(Instances data)
            throws Exception {

        Instances[] localInstances;
        double cost = 0;
        int i;

        if (m_isLeaf) {
            return getCostForDistribution(new Distribution(data));
        } else {
            Distribution savedDist = localModel().distribution();
            localModel().resetDistribution(data);
            localInstances = (Instances[]) localModel().split(data);
            localModel().setDistribution(savedDist);
            for (i = 0; i < m_sons.length; i++) {
                cost = cost
                        + son(i).getCostForBranch(localInstances[i]);
            }
            return cost;
        }
    }

    /**
     * Computes estimated errors for leaf.
     *
     * @param theDistribution the distribution to use
     * @return the estimated errors
     */
    private double getEstimatedErrorsForDistribution(Distribution theDistribution) {

        if (Utils.eq(theDistribution.total(), 0)) {
            return 0;
        } else {
            //TO DO change this to be based on cost
            return theDistribution.numIncorrect()
                    + Stats.addErrs(theDistribution.total(),
                            theDistribution.numIncorrect(), m_CF);
        }
    }
    
    private double getCostForDistribution(Distribution theDistribution) throws Exception {

        if (Utils.eq(theDistribution.total(), 0)) {
            return 0;
        } else {
            // TODO cost sensitive pruning
            // getEstimatedErrorsForDistribution(theDistribution) = Pr*N = expectedErrors
            // theDistribution.total() - getEstimatedErrorsForDistribution(theDistribution) = N - Pr*N = expectedNonErrors
            
            double expectedNonErrors = theDistribution.total() - getEstimatedErrorsForDistribution(theDistribution);
            int thisDistributionPredictionIndex = theDistribution.maxClass();
            double correctCost = m_costMatrix.getElement(thisDistributionPredictionIndex, thisDistributionPredictionIndex) * expectedNonErrors;
            double expectedErrors = getEstimatedErrorsForDistribution(theDistribution);
            double totalErrors = theDistribution.numIncorrect();
            double incorrectCost = 0;
            for(int i = 0; i < theDistribution.numClasses(); i++) {
                if(i != thisDistributionPredictionIndex) {
                    
                    incorrectCost += (theDistribution.perClass(i) / totalErrors) 
                            * expectedErrors //gets which percentage of the "expected error" is this class
                            * m_costMatrix.getElement(thisDistributionPredictionIndex, i);
                    
                }
            }
            //get ratio of 
            
            
            return correctCost + incorrectCost;
        }
    }


    /**
     * Computes errors of tree on training data.
     *
     * @return the training errors
     */
    private double getTrainingErrors() {

        double errors = 0;
        int i;

        if (m_isLeaf) {
            return localModel().distribution().numIncorrect();
        } else {
            for (i = 0; i < m_sons.length; i++) {
                errors = errors + son(i).getTrainingErrors();
            }
            return errors;
        }
    }

    /**
     * Method just exists to make program easier to read.
     *
     * @return the local split model
     */
    private ClassifierSplitModel localModel() {

        return (ClassifierSplitModel) m_localModel;
    }

    /**
     * Computes new distributions of instances for nodes in tree.
     *
     * @param data the data to compute the distributions for
     * @throws Exception if something goes wrong
     */
    private void newDistribution(Instances data) throws Exception {

        Instances[] localInstances;

        localModel().resetDistribution(data);
        m_train = data;
        if (!m_isLeaf) {
            localInstances
                    = (Instances[]) localModel().split(data);
            for (int i = 0; i < m_sons.length; i++) {
                son(i).newDistribution(localInstances[i]);
            }
        } else // Check whether there are some instances at the leaf now!
        if (!Utils.eq(data.sumOfWeights(), 0)) {
            m_isEmpty = false;
        }
    }

    /**
     * Method just exists to make program easier to read.
     */
    private CSTreePruneableClassifierTree son(int index) {

        return (CSTreePruneableClassifierTree) m_sons[index];
    }

    /**
     * Returns the revision string.
     *
     * @return	the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11006 $");
    }
}
