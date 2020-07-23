/*  Implementation of CSForest and CSVoting
    Copyright (C) <2018>  <Michael Furner>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
    
    Author contact details: 
    Name: Michael Furner
    Email: mfurner@csu.edu.au
    Location: 	School of Computing and Mathematics, Charles Sturt University,
    			Bathurst, NSW, Australia, 2795.
 */
package weka.classifiers.trees;

import weka.classifiers.trees.j48.CSTree;
import java.io.Serializable;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.CSTreeSplit;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * <!-- globalinfo-start -->
 * Implementation of the cost-sensitive decision forest algorithm CSForest,
 * which was published in:<br>
 * <br>
 * Siers, M. J., & Islam, M. Z. (2015). Software defect prediction using a cost
 * sensitive decision forest and voting, and a potential solution to the class
 * imbalance problem. Information Systems, 51, 62-71..<br>
 * <br>
 * This cost-sensitive decision forest was originally designed for software
 * defect prediction datasets with two class values ("defective" and
 * "not-defective"). It has been extended here to work with an arbitrary number
 * of class values. The structure of the program is taken from our previous
 * implementation of SysFor, upon which CSForest was based (and which in turn
 * had been based on the Weka implementation of MetaCost). As this uses code
 * from the SysFor implementation in Weka, it is worth noting that the algorithm
 * does not search for "good" attributes beyond the second level of the tree.
 *
 * For classification, the CSForest uses CSVoting, which is specified in the
 * original paper.
 *
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -L &lt;minimum records in leaf&gt;
 *  Set minimum number of records for a leaf.
 *  (default 10)
 * </pre>
 *
 * <pre>
 * -N &lt;no. trees&gt;
 *  Set number of trees to build.
 *  (default 60)
 * </pre>
 *
 * <pre>
 * -G &lt;goodness threshold&gt;
 *  Set goodness threshold for attribute selection.
 *  (default 0.3)
 * </pre>
 *
 * <pre>
 * -S &lt;separation threshold&gt;
 *  Set separation threshold for split point selection.
 *  (default 0.3)
 * </pre>
 *
 * <pre>
 * -C &lt;confidence factor&gt;
 *  Set confidence for pruning.
 *  (default 0.25)
 * </pre>
 * 
 * <pre>
* -A
*  Whether to calculate the total classification cost of the training dataset.
*  (default false)
* </pre>
 *
 * <pre> -cost-matrix &lt;matrix&gt;
 *  The cost matrix in Matlab single line format.
 *  Expanded cost matrix takes form:
 *                    Actual class
 *                         | |
 *                         v v
 *                         a b
 *    class to be   =>  a  0 5
 *  classified as   =>  b  1 1
 * </pre>
 *
 * <!-- options-end -->
 *
 *
 * @author Michael Furner (mfurner@csu.edu.au)
 * @version $Revision: 1.0$
 */
public class CSForest extends AbstractClassifier {

    /**
     * For serialization.
     */
    private static final long serialVersionUID = -5891220800957072995L;

    private boolean correctlySetCostMatrix = true;

    /**
     * The trees that comprise the SysFor forest.
     */
    private ArrayList<Classifier> forest;

    /**
     * Dataset the forest is built on.
     */
    private Instances dataset;

    /**
     * The minimum number of records in a leaf for the C4.5 trees. (default 10)
     */
    private int minRecLeaf = 10;

    /**
     * Display total misclassification cost at end of tree. (default false)
     */
    private boolean calculateTotalCost = false;

    /**
     * The number of trees that the user has requested. In most cases, this
     * number of trees will be created. However, in rare cases, a smaller number
     * is created.
     */
    private int numberTrees = 60;

    /**
     * Used to control the minimum cost ratio for an attribute to be considered
     * for the set of "good attributes". (default 0.2)
     */
    private float costGoodness = 0.2f;

    /**
     * Used to control whether or not a split point may be added to the set of
     * "good attributes" if the split point's attribute is already used in the
     * set of "good attributes". The smaller this value is, the more split
     * points may be used on the same attribute. (default 0.3)
     */
    private float separation = 0.3f;

    /**
     * The confidence factor that will be used in the C4.5 trees. (default 0.25)
     */
    private float confidence = 0.25f;

    /**
     * The number of classes in dataset.
     */
    private int numClasses = -1;

    /**
     * The class names which are used in the string output.
     */
    private String[] classNames;

    /**
     * A variable that is used to store the attribute domains of the passed
     * dataset.
     */
    private double[] attDomains;

    /**
     * The cost matrix
     */
    protected CostMatrix m_CostMatrix = new CostMatrix(2);

    /**
     * CSForest constructor. Makes the cost matrix the default cost matrix from
     * the original paper.
     */
    public CSForest() {
        try {
            String cost_matrix = "[0 5; 1 1]";
            StringWriter writer = new StringWriter();
            CostMatrix.parseMatlab(cost_matrix).write(writer);
            setCostMatrix(new CostMatrix(new StringReader(writer.toString())));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * This method corresponds to Algorithm 1 in the CSForest paper.
     *
     * @param data - data with which to build the classifier
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        getCapabilities().testWithFail(data);
        data = new Instances(data);

        dataset = data;

        /* if the cost matrix hasn't been properly defined, we use a hollow
        matrix with all non-diagonal values as 1 */
        if (m_CostMatrix.size() != dataset.numClasses()) {
            correctlySetCostMatrix = false;
            m_CostMatrix = new CostMatrix(dataset.numClasses());
            for (int i = 0; i < m_CostMatrix.size(); i++) {
                for (int j = 0; j < m_CostMatrix.size(); j++) {
                    if (i == j) {
                        m_CostMatrix.setElement(i, j, 0);
                    } else {
                        m_CostMatrix.setElement(i, j, 1);
                    }
                }
            }
        }

        numClasses = data.numClasses();
        classNames = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classNames[i] = data.classAttribute().value(i);
        }

        attDomains = new double[data.numAttributes()];
        for (int i = 0; i < attDomains.length; i++) {
            attDomains[i] = calculateAttributeDomain(data, i);
        }

        // Remove the records with missing values.
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                data.deleteWithMissing(i);
            }
        }

        //if this is a dataset with only the class attribute
        if (data.numAttributes() == 1) {
            forest = new ArrayList<Classifier>();
            J48 onlyClass = new J48();
            onlyClass.setConfidenceFactor(confidence);
            onlyClass.setMinNumObj(minRecLeaf);
            onlyClass.buildClassifier(data);
            forest.add(onlyClass);
            return;
        }

        // Initialize the forest, good attributes, and split points to empty array lists.
        forest = new ArrayList<Classifier>();
        ArrayList<Attribute> goodAttributes = new ArrayList<Attribute>();
        ArrayList<Double> splitPoints = new ArrayList<Double>();

        ArrayList<GoodAttribute> goodAttributeObjects = getGoodAttributes(data, costGoodness);
        for (int i = 0; i < goodAttributeObjects.size(); i++) {
            goodAttributes.add(goodAttributeObjects.get(i).getAttribute());
            splitPoints.add(goodAttributeObjects.get(i).getSplitPoint());
        }

        int i = 0;
        while ((forest.size() < numberTrees) && (forest.size() < goodAttributes.size())) {
            Attribute currentSplitAttribute = goodAttributes.get(i);
            double currentSplitValue = splitPoints.get(i);
            CSForestTree currentTree = new CSForestTree(new GoodAttribute(currentSplitAttribute, currentSplitValue));
            currentTree.buildClassifier(data);
            forest.add(currentTree);
            i++;
        }
        i = 0;
        int K = forest.size() - 1;

        while ((forest.size() < numberTrees) && (i <= K)) {
            Instances[] dataSplits = splitData(data, goodAttributeObjects.get(i));
            ArrayList<ArrayList<Attribute>> levelTwoGoodAttributes = new ArrayList<ArrayList<Attribute>>();
            ArrayList<ArrayList<Double>> levelTwoSplitPoints = new ArrayList<ArrayList<Double>>();
            ArrayList<ArrayList<GoodAttribute>> levelTwoGoodAttributeObjects = new ArrayList<ArrayList<GoodAttribute>>();

            for (int j = 0; j < dataSplits.length; j++) {
                ArrayList<Attribute> currentSplitGoodAttributes = new ArrayList<Attribute>();
                ArrayList<Double> currentSplitPoints = new ArrayList<Double>();

                ArrayList<GoodAttribute> currentGoodAttributeObjects = getGoodAttributes(dataSplits[j], costGoodness);
                for (int l = 0; l < currentGoodAttributeObjects.size(); l++) {
                    currentSplitGoodAttributes.add(currentGoodAttributeObjects.get(l).getAttribute());
                    currentSplitPoints.add(currentGoodAttributeObjects.get(l).getSplitPoint());
                }
                levelTwoGoodAttributeObjects.add(currentGoodAttributeObjects);

                levelTwoGoodAttributes.add(currentSplitGoodAttributes);
                levelTwoSplitPoints.add(currentSplitPoints);
            }

            // Calculate the possible number of trees.
            // Here it is broken down into numerator and denominator for readability.
            int possibleNumberTrees = 0;
            int numerator = 0;
            int denominator = 0;

            // Calculate the numerator and denominator
            for (int j = 0; j < dataSplits.length; j++) {
                numerator += levelTwoGoodAttributes.get(j).size() * dataSplits[j].numInstances();
                denominator += dataSplits[j].numInstances();
            }

            possibleNumberTrees = numerator / denominator;

            int x = 0;
            ArrayList<CSForestTree> levelTwoTrees = null;
            while ((forest.size() < numberTrees) && (x <= possibleNumberTrees - 1)) {
                levelTwoTrees = new ArrayList<CSForestTree>();
                for (int j = 0; j < dataSplits.length; j++) {
                    if ((levelTwoGoodAttributes.get(j).size() - 1) > x) {
                        CSForestTree newSubTree = new CSForestTree(levelTwoGoodAttributeObjects.get(j).get(x + 1));
                        newSubTree.buildClassifier(dataSplits[j]);
                        levelTwoTrees.add(newSubTree);
                    } else {
                        CSForestTree newSubTree;
                        // There won't be any good attribute objects if there's only 1 record in the split.
                        if (dataSplits[j].numInstances() == 1 || levelTwoGoodAttributeObjects.get(j).isEmpty()) {
                            newSubTree = new CSForestTree();
                        } else {
                            newSubTree = new CSForestTree(levelTwoGoodAttributeObjects.get(j).get(0));
                        }
                        newSubTree.buildClassifier(dataSplits[j]);
                        levelTwoTrees.add(newSubTree);
                    }
                }
                CSForestTree levelTwoTreesArray[] = new CSForestTree[levelTwoTrees.size()];
                for (int j = 0; j < levelTwoTreesArray.length; j++) {
                    levelTwoTreesArray[j] = levelTwoTrees.get(j);
                }
                CSForestLevelTwoTree levelTwoTree = new CSForestLevelTwoTree(levelTwoTreesArray,
                        goodAttributeObjects.get(i));
                levelTwoTree.buildClassifier(data);
                forest.add(levelTwoTree);
                x++;
            }
            i++;
        }

    }

    /**
     * This method corresponds to Algorithm 2 in the SysFor paper.
     *
     * @param dataset - subsection of the dataset on which to find good
     * attributes
     * @param goodness - goodness threshold, determines the maximum difference
     * between gain ratios of selected split points
     * @return the good attributes
     * java.util.ArrayList<weka.classifiers.trees.SysFor.GoodAttribute>
     */
    private ArrayList<GoodAttribute> getGoodAttributes(Instances dataset, float goodnessThreshold) throws Exception {

        //calculate total expected cost
        double totalExpectedCost = 0;
        double[] fullCostArray = new double[dataset.classAttribute().numValues()];
        for (int x = 0; x < dataset.classAttribute().numValues(); x++) {

            //for each of the possible class values we need to calculate 
            //classification and misclassification costs
            String thisClassValue = dataset.classAttribute().value(x);

            double runningSum = 0;

            for (int y = 0; y < dataset.classAttribute().numValues(); y++) {

                RemoveWithValues rmv = new RemoveWithValues();
                int classIdx = dataset.classIndex() + 1;
                rmv.setAttributeIndex("" + classIdx);
                int[] indicesArr = {y};
                rmv.setNominalIndicesArr(indicesArr);
                rmv.setInvertSelection(true);
                rmv.setInputFormat(new Instances(dataset));

                Instances temp = Filter.useFilter(new Instances(dataset), rmv); //temp dataset with only this class j
                int numInstancesOfJ = temp.numInstances();

                double c_ij = m_CostMatrix.getElement(x, y) * numInstancesOfJ;
                runningSum += c_ij;

            }

            fullCostArray[x] = runningSum;

        }

        double numerator = 2;
        double denominator = 0;
        for (int j = 0; j < fullCostArray.length; j++) {
            numerator *= fullCostArray[j];
            denominator += fullCostArray[j];
        }
        totalExpectedCost = numerator / denominator;
        /**
         * ************ finished calculating totalExpectedCost ************
         */

        // Initialize a set of attributes, a set of split points, and a set of gain ratios to empty.
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        ArrayList<Double> splitPoints = new ArrayList<Double>();
        ArrayList<Double> costs = new ArrayList<Double>();

        // For each attribute in the dataset (that isn't the class attribute):
        for (int i = 0; i < dataset.numAttributes(); i++) {
            // Calculate the domain of the attribute (highest value minus lowest value)
            double currentAttributeDomain = attDomains[i];

            Attribute currentAttribute = dataset.attribute(i);
            if (i == dataset.classIndex()) {
                continue;
            }
            // If the current attribute is categorical:
            if (currentAttribute.isNominal()) {
                // Calculate the cost of the current attribute.
                CSTreeSplit split = new CSTreeSplit(i, minRecLeaf, 0.0, m_CostMatrix, totalExpectedCost);
                split.buildClassifier(dataset);

                // Store the corresponding information into attributes, splitPoints, and gainRatios
                Distribution dist = split.distribution();
                int[][] perSplitClassCount = new int[dist.numBags()][dataset.numClasses()];
                int splitsThatHaveMinimumRecords = 0;
                for (int x = 0; x < perSplitClassCount.length; x++) {
                    for (int y = 0; y < perSplitClassCount[x].length; y++) {
                        perSplitClassCount[x][y] = (int) dist.perClassPerBag(x, y);
                    }
                    if (dist.perBag(x) >= minRecLeaf) {
                        splitsThatHaveMinimumRecords++;
                    }
                }

                if (splitsThatHaveMinimumRecords < 2) {
                    continue;
                }

                Distribution noSplitDist = null;
                noSplitDist = new Distribution(dataset);

                //Instead of "information before splitting" we use the totalExpectedCost
                double cost = split.cost(); //calculateGainNominal(perSplitClassCount, dataset.numClasses(), totalExpectedCost);
                costs.add(cost);
                splitPoints.add(-100.0);
                attributes.add(currentAttribute);
            } else if (currentAttribute.isNumeric()) {
                // Initialize a set of candidate attributes 
                // and a set of candidate costs to empty.
                ArrayList<Double> candidateSplitPoints = new ArrayList<Double>();
                ArrayList<Double> candidateCosts = new ArrayList<Double>();

                // Create a set of available split points on currentAttribute
                ArrayList<Double> availableSplitPoints = findAvailableSplitPoints(dataset, i);
                // Create a set of corresponding costs
                ArrayList<Double> availableCosts = new ArrayList<Double>();
                for (int j = 0; j < availableSplitPoints.size(); j++) {
                    availableCosts.add(0.0);
                }

                // Move instances from the right side of a potential split one split point at a time.
                // During each iteration, calculate the cost of the split and store it in availableCosts 
                availableCosts = calculateNumericSplitsCost(dataset, availableSplitPoints, i, totalExpectedCost);

                // Find all the best split points within this attribute that satisfy the separation threshold.
                while (!availableSplitPoints.isEmpty()) {
                    // Find the highest cost reduction from the available cost reductions.
                    double maxCostReduction = Double.NEGATIVE_INFINITY;
                    int maxCostReductionIndex = -1;
                    for (int j = 0; j < availableCosts.size(); j++) {
                        if (availableCosts.get(j) > maxCostReduction) {
                            maxCostReduction = availableCosts.get(j);
                            maxCostReductionIndex = j;
                        }
                    }

                    // Store the highest gain ratio and its corresponding split point in candidateGainRatios 
                    // and candidateSplitPoints, then remove them from the available lists.
                    candidateCosts.add(availableCosts.get(maxCostReductionIndex));
                    candidateSplitPoints.add(availableSplitPoints.get(maxCostReductionIndex));
                    availableCosts.remove(maxCostReductionIndex);
                    availableSplitPoints.remove(maxCostReductionIndex);

                    ArrayList<Integer> newAvailableSplitPointsIndices = recalculateAvailableSplitPoints(
                            availableSplitPoints,
                            candidateSplitPoints,
                            currentAttributeDomain);

                    // Update availableCosts and availableSplitPoints using the above recalculated indices
                    ArrayList<Double> newAvailableSplitPoints = new ArrayList<Double>();
                    ArrayList<Double> newAvailableCosts = new ArrayList<Double>();
                    for (int j = 0; j < newAvailableSplitPointsIndices.size(); j++) {
                        newAvailableSplitPoints.add(availableSplitPoints.get(newAvailableSplitPointsIndices.get(j)));
                        newAvailableCosts.add(availableCosts.get(newAvailableSplitPointsIndices.get(j)));
                    }
                    availableSplitPoints = newAvailableSplitPoints;
                    availableCosts = newAvailableCosts;
                }

                while (!candidateSplitPoints.isEmpty()) {
                    attributes.add(currentAttribute);
                    // Find the lowest cost from the candidate costs.

                    double maxCostReduction = Double.NEGATIVE_INFINITY;
                    int maxCostReductionIndex = -1;
                    for (int j = 0; j < candidateCosts.size(); j++) {
                        if (candidateCosts.get(j) > maxCostReduction) {
                            maxCostReduction = candidateCosts.get(j);
                            maxCostReductionIndex = j;
                        }
                    }
                    costs.add(maxCostReduction);
                    splitPoints.add(candidateSplitPoints.get(maxCostReductionIndex));

                    // Remove the best cost and corresponding split point from candidateCosts and 
                    // candidateSplitPoints respectively.
                    candidateCosts.remove(maxCostReductionIndex);
                    candidateSplitPoints.remove(maxCostReductionIndex);
                }
            }
        }
        // Sort the costs in ascending order. Update the attributes and splitpoints accordingly.
        GoodAttributesWithCostReductions goodAttributesWithCR = new GoodAttributesWithCostReductions(attributes,
                splitPoints,
                costs);
        goodAttributesWithCR.sort();

        // Remove the elements in costs, and corresponding elements in attributes and splitPoints
        // where the difference between a cost and the best cost in costs is greater than 
        // goodness
        if (goodAttributesWithCR.size() > 0) {
            double bestCost = goodAttributesWithCR.getCostReduction(0);
            for (int i = 1; i < goodAttributesWithCR.size(); i++) {
                double currentCost = goodAttributesWithCR.getCostReduction(i);
                if (currentCost == Double.NEGATIVE_INFINITY) {
                    currentCost = 0;
                }

                double costDist = 1.0 - currentCost / bestCost;
                if (costDist < goodnessThreshold) {
                    goodAttributesWithCR.remove(i);
                    i--;
                }
            }
        }

        return goodAttributesWithCR.getGoodAttributes();
    }

    /**
     * Returns the distribution of tree votes for the available classes,
     * classifying using CSVoting.
     *
     * @param instance - the instance to be classified
     * @return probablity distribution for this instance's classification
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws java.lang.Exception {

        //if the forest hasn't been built or is empty due to poorly selected attribute values
        if (forest == null || forest.isEmpty()) {
            ZeroR zr = new ZeroR();
            zr.buildClassifier(dataset);
            return zr.distributionForInstance(instance);
        }

        //initialise cost array for this instance
        double[] returnValue = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            returnValue[i] = 0;
        }

        for (int j = 0; j < forest.size(); j++) {

            //get the distribution of costs for this instance
            double[] currentTreeDistribution = forest.get(j).distributionForInstance(instance);
            for (int i = 0; i < numClasses; i++) {
                returnValue[i] += currentTreeDistribution[i];
            }

        }

        //turn the costs into the weka-expected distribution of class probabilities
        returnValue = CSTree.invertCosts(returnValue);

        return returnValue;

    }

    /**
     * Calculate the domain of the attribute (highest value minus lowest value
     * for numerical. Number of distinct values for nominal attributes).
     *
     * @param dataset - dataset on which to calculate the domain
     * @param attributeIndex - index in the dataset of the attribute to
     * calculate
     * @return the range of the dataset for numerical attributes, number of
     * distinct values for nominal attributes
     */
    private double calculateAttributeDomain(Instances dataset, int attributeIndex) {
        if (dataset.attribute(attributeIndex).isNumeric()) {
            // Sort the attributes values and store them as a list.
            double[] values = dataset.attributeToDoubleArray(attributeIndex);
            Double[] objectValues = new Double[values.length];
            for (int i = 0; i < objectValues.length; i++) {
                objectValues[i] = (Double) values[i];
            }
            Arrays.sort(objectValues);
            ArrayList<Double> lValues = new ArrayList<Double>();
            lValues.addAll(Arrays.asList(objectValues));

            // Create an iterator object for the sorted attribute values
            Iterator<Double> iValues = lValues.iterator();

            double minValue = Double.POSITIVE_INFINITY;
            double maxValue = Double.NEGATIVE_INFINITY;

            while (iValues.hasNext()) {
                double currentValue = (Double) iValues.next();
                if (currentValue < minValue) {
                    minValue = currentValue;
                }
                if (currentValue > maxValue) {
                    maxValue = currentValue;
                }
            }
            double domain = maxValue - minValue;
            return domain;
        } else {
            return dataset.numDistinctValues(attributeIndex);
        }

    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain the following arguments: -t training file [-T
     * test file] [-c class index]
     */
    public static void main(String[] argv) {
        runClassifier(new CSForest(), argv);
    }

    /**
     * Returns capabilities of algorithm
     *
     * @return Weka capabilities of SysFor
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.disable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.STRING_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.disable(Capabilities.Capability.NUMERIC_CLASS);
        result.disable(Capabilities.Capability.DATE_CLASS);
        result.disable(Capabilities.Capability.RELATIONAL_CLASS);
        result.disable(Capabilities.Capability.UNARY_CLASS);
        result.disable(Capabilities.Capability.NO_CLASS);
        result.disable(Capabilities.Capability.STRING_CLASS);
        return result;

    }

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {
        return "Implementation of the cost sensitive decision forest algorithm CSForest, which was published in: \n"
                + "Siers, M. J., & Islam, M. Z. (2015). Software defect prediction using a cost sensitive decision "
                + "forest and voting, and a potential solution to the class imbalance problem.\n"
                + "For more information, see:\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Inner class for GoodAttributes. A GoodAttribute object is comprised of an
     * attribute, and a split point on that attribute.
     */
    private class GoodAttribute implements Serializable {

        private static final long serialVersionUID = 5731302547322418250L;

        protected Attribute attribute;
        protected double splitPoint;

        public GoodAttribute(Attribute a, double split) {
            this.setAttribute(a, split);
        }

        protected void setAttribute(Attribute a, double split) {
            attribute = a;
            splitPoint = split;
        }

        protected Attribute getAttribute() {
            return this.attribute;
        }

        protected Double getSplitPoint() {
            return splitPoint;
        }
    }

    /**
     * This class is used specifically to store attribute, splitPoint, and
     * corresponding cost information in one object. It is used in the
     * GoodAttributesWithCosts class which can sort a collection of
     * GoodAttributeWithCost objects.
     *
     * @author Michael Furner, based on code from Michael J. Siers
     */
    private class GoodAttributeWithCostReduction extends GoodAttribute implements Comparable {

        private static final long serialVersionUID = 3700968715919025916L;

        private double costReduction;

        public double getCostReduction() {
            return costReduction;
        }

        public GoodAttributeWithCostReduction(Attribute a, double split, double costReduction) {
            super(a, split);
            this.costReduction = costReduction;
        }

        @Override
        public int compareTo(Object a) {
            return Double.compare(this.costReduction, ((GoodAttributeWithCostReduction) a).costReduction);
        }
    }

    /**
     * This class is used specifically to sort attribute, splitPoint, and cost
     * arrays simultaneously based on costs.
     *
     * @author Michael J. Siers
     */
    private class GoodAttributesWithCostReductions implements Serializable {

        private static final long serialVersionUID = 2339265760515386479L;

        private ArrayList<GoodAttributeWithCostReduction> elements;

        public GoodAttributesWithCostReductions(ArrayList<Attribute> attributes, ArrayList<Double> splitPoints,
                ArrayList<Double> costs) {

            elements = new ArrayList<GoodAttributeWithCostReduction>();
            for (int i = 0; i < attributes.size(); i++) {
                elements.add(new GoodAttributeWithCostReduction(attributes.get(i), splitPoints.get(i), costs.get(i)));
            }
        }

        /**
         * Sorts the elements into ascending order of cost
         */
        public void sort() {
            Collections.sort(elements, Collections.reverseOrder());
        }

        public ArrayList<GoodAttribute> getGoodAttributes() {
            ArrayList<GoodAttribute> returnValue = new ArrayList<GoodAttribute>();

            for (int i = 0; i < elements.size(); i++) {
                returnValue.add(elements.get(i));
            }

            return returnValue;
        }

        public int size() {
            return elements.size();
        }

        public double getCostReduction(int index) {
            return elements.get(index).costReduction;
        }

        public void remove(int index) {
            elements.remove(index);
        }
    }

    private ArrayList<Double> findAvailableSplitPoints(Instances dataset, int attributeIndex) {
        // Initialize the return value
        ArrayList<Double> splitPoints = new ArrayList<Double>();
        if (dataset.numInstances() == 0 || dataset.numInstances() == 1) {
            return splitPoints;
        }

        // Sort the attributes values and store them as a list.
        double[] values = dataset.attributeToDoubleArray(attributeIndex);
        Double[] objectValues = new Double[values.length];
        for (int i = 0; i < objectValues.length; i++) {
            objectValues[i] = (Double) values[i];
        }
        Arrays.sort(objectValues);
        ArrayList<Double> lValues = new ArrayList<Double>();
        lValues.addAll(Arrays.asList(objectValues));

        // Create an iterator object for the sorted attribute values
        Iterator<Double> iValues = lValues.iterator();

        // Get the first element
        double previousElement = iValues.next();
        NumericSplitDistribution dist = new NumericSplitDistribution(dataset, attributeIndex);

        // Now add the midpoints between each adjacent value to the return value.
        do {
            double valueOne = previousElement;
            double valueTwo = (Double) iValues.next();

            if (valueOne != valueTwo) {
                double midPointValue = (valueOne + valueTwo) / 2;

                dist.shiftRecords(midPointValue);
                if (dist.getNumberLeftSideInstances() >= minRecLeaf && dist.getNumberRightSideInstances() >= minRecLeaf) {
                    splitPoints.add(midPointValue);
                }
            }

            previousElement = valueTwo;
        } while (iValues.hasNext());

        return splitPoints;
    }

    /**
     * This method removes split points from the passed array which do not
     * satisfy the separation threshold equation.
     *
     * @param availableSplitPoints the available split points
     * @param candidateSplitPoints the already candidate split points
     * @param attributeDomain the highest value of this attribute minus the
     * lowest attribute.
     * @return the indexes of the recalculated available split points
     */
    private ArrayList<Integer> recalculateAvailableSplitPoints(ArrayList<Double> availableSplitPoints,
            ArrayList<Double> candidateSplitPoints,
            double attributeDomain) {

        ArrayList<Integer> returnValue = new ArrayList<Integer>();

        for (int i = 0; i < availableSplitPoints.size(); i++) {
            double currentSplitPoint = availableSplitPoints.get(i);
            boolean keepCurrentPoint = true;
            for (int j = 0; j < candidateSplitPoints.size(); j++) {
                if ((Math.abs(currentSplitPoint - candidateSplitPoints.get(j))) / attributeDomain <= separation) {
                    keepCurrentPoint = false;
                    break;
                }
            }
            if (keepCurrentPoint) {
                returnValue.add(i);
            }
        }

        return returnValue;
    }

    /**
     * A method for finding the costs for each of the split points of a numeric
     * attribute. The method returns a list of costs.
     *
     * @param instances
     * @param availableSplitPoints
     * @param attrIndex
     * @param noSplitCost
     * @return split cost for each of the split points
     * @throws java.lang.Exception if there are issues with the cost matrix
     */
    public ArrayList<Double> calculateNumericSplitsCost(Instances instances, ArrayList<Double> availableSplitPoints, int attrIndex, double noSplitCost) throws Exception {
        ArrayList<Double> availableCosts = new ArrayList<Double>();
        for (int i = 0; i < availableSplitPoints.size(); i++) {
            availableCosts.add(0.0);
        }
        // Calculate the cost before splitting.
        NumericSplitDistribution dist = new NumericSplitDistribution(instances, attrIndex);
        int totalLeft = dist.leftSideInstances.numInstances();
        int totalRight = dist.rightSideInstances.numInstances();

        double cost = 0.0;
        for (int i = 0; i < availableSplitPoints.size(); i++) {
            dist.shiftRecords(availableSplitPoints.get(i));
            cost = calculateCostReductionNumeric(dist.leftClassCount, dist.rightClassCount, instances.numClasses(), noSplitCost);
            availableCosts.set(i, cost);
        }
        return availableCosts;
    }

    /**
     * Calculates the split info given number of class instances each side of
     * split for a numeric split
     *
     * @param leftClassCount - class instances in left of split
     * @param rightClassCount - class instances in right of split
     * @param numClasses - amount of classes in dataset
     * @return splitInfo for given numeric split
     */
    /* public double calculateNumericSplitInfo(int[] leftClassCount, int[] rightClassCount, int numClasses) {
        double splitInfo = 0.0;

        int totalRight = 0;
        int totalLeft = 0;
        for (int i = 0; i < numClasses; i++) {
            totalRight += rightClassCount[i];
            totalLeft += leftClassCount[i];
        }
        int totalRecords = totalRight + totalLeft;

        double x1 = (double) totalLeft / totalRecords;
        double y1 = logFunc((double) totalLeft / totalRecords);
        double z1 = x1 * y1;
        double x2 = (double) totalRight / totalRecords;
        double y2 = logFunc((double) totalRight / totalRecords);
        double z2 = x2 * y2;

        splitInfo += ((double) totalLeft / totalRecords) * logFunc((double) totalLeft / totalRecords);
        splitInfo += ((double) totalRight / totalRecords) * logFunc((double) totalRight / totalRecords);

        return -splitInfo;
    } */
    /**
     * Calculates the split info given number of class instances each part of a
     * nominal split
     *
     * @param dataSplitsClassCounts - Class counts in each of the subsets
     * created in the nominal split
     * @param numClasses - number of classes in the dataset
     * @return
     */
    /*public double calculateNominalSplitInfo(int[][] dataSplitsClassCounts, int numClasses) {
        double splitInfo = 0.0;

        int[] totalSplitCounts = new int[dataSplitsClassCounts.length];
        for (int j = 0; j < dataSplitsClassCounts.length; j++) {
            for (int i = 0; i < numClasses; i++) {
                totalSplitCounts[j] += dataSplitsClassCounts[j][i];
            }
        }

        int totalRecords = 0;
        for (int i = 0; i < totalSplitCounts.length; i++) {
            totalRecords += totalSplitCounts[i];
        }

        for (int i = 0; i < dataSplitsClassCounts.length; i++) {
            splitInfo += ((double) totalSplitCounts[i] / totalRecords) * logFunc((double) totalSplitCounts[i] / totalRecords);
        }

        return -splitInfo;
    } */
    /**
     * Calculates the cost for a split on a numerical attribute
     *
     * @param leftClassCount - class instances on left of split
     * @param rightClassCount - class instances on right of split
     * @param numClasses - number of distinct classes in dataset
     * @param costBeforeSplit - cost prior to split
     * @return cost reduction
     * @throws java.lang.Exception if there is a problem with the cost matrix
     */
    public double calculateCostReductionNumeric(int[] leftClassCount, int[] rightClassCount, int numClasses, double costBeforeSplit) throws Exception {
        double costAfterSplit = 0.0;

        int totalRight = 0;
        int totalLeft = 0;
        for (int i = 0; i < numClasses; i++) {
            totalRight += rightClassCount[i];
            totalLeft += leftClassCount[i];
        }
        int totalRecords = totalRight + totalLeft;

        double costLessThan = 0.0;
        double costGreaterThan = 0.0;

        //calculate cost for left side
        double[] costPerClassArray = new double[numClasses];

        for (int classToBeClassifiedAs = 0; classToBeClassifiedAs < numClasses; classToBeClassifiedAs++) {
            double runningSum = 0;

            for (int actualClassNumbersIdx = 0; actualClassNumbersIdx < numClasses; actualClassNumbersIdx++) {

                double c_ij = m_CostMatrix.getElement(classToBeClassifiedAs, actualClassNumbersIdx) * leftClassCount[actualClassNumbersIdx];
                runningSum += c_ij;

            }

            costPerClassArray[classToBeClassifiedAs] = runningSum;

        }

        double numerator = 1;
        double denominator = 0;
        for (int k = 0; k < costPerClassArray.length; k++) {
            numerator *= costPerClassArray[k];
            denominator += costPerClassArray[k];
        }
        costLessThan = 2 * numerator / denominator;

        //calculate cost for right side
        costPerClassArray = new double[numClasses];

        for (int classToBeClassifiedAs = 0; classToBeClassifiedAs < numClasses; classToBeClassifiedAs++) {
            double runningSum = 0;

            for (int actualClassNumbersIdx = 0; actualClassNumbersIdx < numClasses; actualClassNumbersIdx++) {

                double c_ij = m_CostMatrix.getElement(classToBeClassifiedAs, actualClassNumbersIdx) * rightClassCount[actualClassNumbersIdx];
                runningSum += c_ij;

            }

            costPerClassArray[classToBeClassifiedAs] = runningSum;

        }

        numerator = 1;
        denominator = 0;
        for (int k = 0; k < costPerClassArray.length; k++) {
            numerator *= costPerClassArray[k];
            denominator += costPerClassArray[k];
        }
        costGreaterThan = 2 * numerator / denominator;

        costAfterSplit = costGreaterThan + costLessThan;

        return costBeforeSplit - costAfterSplit;
    }

    /**
     * Calculates the cost reduction for a split on a nominal attribute
     *
     * @param dataSplitsClassCounts - class counts for the subsets from the
     * split
     * @param numClasses - number of distinct classes in dataset
     * @param costBeforeSplit - cost prior to split
     * @return
     * @throws java.lang.Exception if there is problem with the cost matrix
     */
    public double calculateCostReductionNominal(int[][] dataSplitsClassCounts, int numClasses, double costBeforeSplit) throws Exception {
        double costAfterSplit = 0.0;

        int[] totalSplitCounts = new int[dataSplitsClassCounts.length];
        for (int j = 0; j < dataSplitsClassCounts.length; j++) {
            for (int i = 0; i < numClasses; i++) {
                totalSplitCounts[j] += dataSplitsClassCounts[j][i];
            }
        }

        int totalRecords = 0;
        boolean fruitlessSplit = false; // if this is a split on an already homogenous dataset
        for (int i = 0; i < totalSplitCounts.length; i++) {
            if (totalSplitCounts[i] == 0) {
                fruitlessSplit = true;
            }
            totalRecords += totalSplitCounts[i];
        }

        if (!fruitlessSplit) {

            double[] splitCosts = new double[dataSplitsClassCounts.length];

            for (int i = 0; i < splitCosts.length; i++) {

                double[] costPerClassArray = new double[numClasses];

                for (int classToBeClassifiedAs = 0; classToBeClassifiedAs < numClasses; classToBeClassifiedAs++) {
                    double runningSum = 0;

                    for (int actualClassNumbersIdx = 0; actualClassNumbersIdx < numClasses; actualClassNumbersIdx++) {

                        double c_ij = m_CostMatrix.getElement(classToBeClassifiedAs, actualClassNumbersIdx) * dataSplitsClassCounts[i][actualClassNumbersIdx];
                        runningSum += c_ij;

                    }

                    costPerClassArray[classToBeClassifiedAs] = runningSum;

                }

                double numerator = 1;
                double denominator = 0;
                for (int k = 0; k < costPerClassArray.length; k++) {
                    numerator *= costPerClassArray[k];
                    denominator += costPerClassArray[k];
                }
                splitCosts[i] = 2 * numerator / denominator;

            }

            costAfterSplit = 0;
            for (int i = 0; i < splitCosts.length; i++) {
                costAfterSplit += splitCosts[i];
            }
        } else { //fruitless split
            costAfterSplit = Double.MAX_VALUE;
        }

        return costBeforeSplit - costAfterSplit;
    }

    /**
     * This class can keep track of how many records of each class are in each
     * side of a numeric CSForest split.
     *
     * @author Michael J. Siers
     *
     */
    private class NumericSplitDistribution implements Serializable {

        private static final long serialVersionUID = 5260380811391105058L;

        private Instances leftSideInstances;
        private Instances rightSideInstances;
        private int[] leftClassCount;
        private int[] rightClassCount;
        private int attrIndex;

        /**
         * Constructor
         *
         * @param instances all of the instances
         * @param attrIndex the index of the attribute that will be tested to
         * create the split
         */
        public NumericSplitDistribution(Instances instances, int attrIndex) {
            this.attrIndex = attrIndex;
            this.leftSideInstances = new Instances(instances, 0);
            this.rightSideInstances = new Instances(instances);
            this.rightSideInstances.sort(attrIndex);
            leftClassCount = new int[instances.numClasses()];
            rightClassCount = new int[instances.numClasses()];

            Enumeration eInstances = instances.enumerateInstances();
            while (eInstances.hasMoreElements()) {
                Instance currentInstance = (Instance) eInstances.nextElement();
                rightClassCount[(int) currentInstance.classValue()]++;
            }
        }

        /**
         * Provides total number of instances at this split point
         *
         * @return number of instances at split point
         */
        public int getNumberInstances() {
            return leftSideInstances.size() + rightSideInstances.size();
        }

        /**
         * Provides number of instances on the right of this split
         *
         * @return number of instances at right of split
         */
        public int getNumberRightSideInstances() {
            return rightSideInstances.size();
        }

        /**
         * Provides number of instances on the left of this split
         *
         * @return number of instances at left of split
         */
        public int getNumberLeftSideInstances() {
            return leftSideInstances.size();
        }

        /**
         * Shifts the records based on the passed split point. Also returns the
         * number of records that have been shifted.
         *
         * @param splitPoint the split value. Records with less than or equal to
         * will be shifted from the right side to the left side.
         * @return the number of records that were shifted.
         */
        public int shiftRecords(double splitPoint) {
            if (rightSideInstances.numInstances() == 0) {
                return 0;
            }
            int shiftCount = 0;

            while (rightSideInstances.numInstances() > 0 && rightSideInstances.firstInstance().value(attrIndex) <= splitPoint) {
                Instance currentInstance = rightSideInstances.firstInstance();
                leftClassCount[(int) currentInstance.classValue()]++;
                rightClassCount[(int) currentInstance.classValue()]--;
                leftSideInstances.add(new DenseInstance(currentInstance));
                rightSideInstances.delete(0);
                shiftCount++;
            }

            return shiftCount;
        }
    }

    /**
     * An inner class for representing a single tree within a CSForest forest.
     * This is a C4.5 tree with a specified root attribute. The split point is
     * also specified if the attribute is numeric. There is another class:
     * "CSForestLevelTwoTree" which is to be used for representing a
     * CSForestTree which has both the root attribute, and some/all of the next
     * level split points specified also.
     *
     * @author Michael Furner, based on code from Michael J. Siers
     */
    private class CSForestTree extends AbstractClassifier {

        private static final long serialVersionUID = -6792080072901419517L;

        private ArrayList<Integer> distribution;
        private int[][] splitsClassDist;
        private double[][] splitsClassCosts;
        private GoodAttribute rootSplit;
        private int majorityIndex;
        private int[] splitsMinCostIndexes;
        //private int[] splitsMajorityIndexes;
        private String[] splitsMinCostValueNames;
        //private String[] splitsMajorityValueNames;
        private ArrayList<CSTree> csTrees = new ArrayList<CSTree>();
        private int numClasses = -1;
        private boolean leaf = false;

        // The number of leaves in this sysfor tree
        private int numberLeaves = 0;

        public CSForestTree(GoodAttribute rootSplit) {
            this.rootSplit = rootSplit;
        }

        /**
         * Creates a CSForestTree object which will not split
         */
        public CSForestTree() {
            leaf = true;
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws java.lang.Exception {
            if (leaf) {
                double[] rootDist = new double[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    rootDist[i] = (double) splitsClassCosts[0][i];
                }
                return CSTree.invertCosts(rootDist);
            }
            double[] returnValue = new double[numClasses];

            Attribute rootAttribute = rootSplit.getAttribute();
            double splitValue = rootSplit.getSplitPoint();

            if (rootAttribute.isNumeric()) {
                if (instance.value(rootAttribute) > splitValue) {
                    if (csTrees.get(0) == null) {
                        for (int i = 0; i < numClasses; i++) {
                            returnValue[i] = 0;
                        }
                        return returnValue;
                    } else {
                        return csTrees.get(0).distributionCosts(instance);
                    }
                } else if (csTrees.get(1) == null) {
                    for (int i = 0; i < numClasses; i++) {
                        returnValue[i] = 0;
                    }
                    return returnValue;
                } else {
                    return csTrees.get(1).distributionCosts(instance);
                }
            } else if (rootAttribute.isNominal()) {
                int treeIndex = (int) instance.value(rootAttribute);
                if (csTrees.get(treeIndex) != null) {
                    return csTrees.get(treeIndex).distributionCosts(instance);
                } else {
                    for (int i = 0; i < numClasses; i++) {
                        returnValue[i] = 0;
                    }
                    return returnValue;
                }

            } else {
                for (int i = 0; i < numClasses; i++) {
                    returnValue[i] = 0;
                }
                return returnValue;
            }
        }

        public int measureNumLeaves() {
            int numLeaves = 0;
            for (int i = 0; i < csTrees.size(); i++) {
                if (csTrees.get(i) == null) {
                    numLeaves++;
                } else {
                    int currentTreeNumLeaves = (int) csTrees.get(i).measureNumLeaves();
                    if (currentTreeNumLeaves == 0) {
                        numLeaves++;
                    } else {
                        numLeaves += currentTreeNumLeaves;
                    }
                }
            }
            return numLeaves;
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            this.numClasses = data.numClasses();
            if (leaf) {
                Instances[] dataSplits = new Instances[1];
                dataSplits[0] = new Instances(data);

                distribution = new ArrayList<Integer>();
                distribution.add(1);
                splitsClassDist = new int[1][numClasses];
                splitsClassCosts = new double[1][numClasses];

                if (data.isEmpty()) {
                    splitsClassDist[0][0]++;
                    splitsMinCostIndexes = new int[1];
                    splitsMinCostIndexes[0] = 0;
                } else {
                    splitsClassDist[0][(int) data.firstInstance().classValue()]++;
                    splitsMinCostIndexes = new int[1];
                    splitsMinCostIndexes[0] = (int) data.firstInstance().classValue();
                }

                for (int i = 0; i < dataSplits.length; i++) {

                    for (int toBeClassifiedAs = 0; toBeClassifiedAs < data.classAttribute().numValues(); toBeClassifiedAs++) {

                        double cost = 0;

                        for (int j = 0; j < data.classAttribute().numValues(); j++) {

                            cost += m_CostMatrix.getElement(toBeClassifiedAs, j) * splitsClassDist[i][j];

                        }

                        splitsClassCosts[i][toBeClassifiedAs] = cost;

                    }

                }

                splitsMinCostValueNames = new String[dataSplits.length];
                splitsMinCostValueNames[0] = data.classAttribute().value(splitsMinCostIndexes[0]);

                for (int i = 0; i < dataSplits.length; i++) {
                    CSTree currentSubTree = new CSTree();
                    currentSubTree.setConfidenceFactor(confidence);
                    currentSubTree.setMinNumObj(minRecLeaf);
                    currentSubTree.setCostMatrix(m_CostMatrix);
                    currentSubTree.setCollapseTree(false); //as this is not used in siers implementation
                    currentSubTree.setSubtreeRaising(false); //as this is not used in siers implementation

                    if (dataSplits[i].numInstances() > 0) {
                        currentSubTree.buildClassifier(dataSplits[i]);
                        csTrees.add(currentSubTree);
                        numberLeaves += currentSubTree.measureNumLeaves();
                    } else {
                        csTrees.add(null);
                        numberLeaves += 1;
                    }
                }

                return;
            }

            Instances[] dataSplits = splitData(data, rootSplit);
            distribution = new ArrayList<Integer>();
            splitsClassDist = new int[dataSplits.length][numClasses];
            splitsClassCosts = new double[dataSplits.length][numClasses];
            splitsMinCostIndexes = new int[dataSplits.length];
            splitsMinCostValueNames = new String[dataSplits.length];

            for (int i = 0; i < dataSplits.length; i++) {
                distribution.add(0);
            }

            for (int i = 0; i < dataSplits.length; i++) {
                distribution.set(i, dataSplits[i].numInstances());
            }

            for (int i = 0; i < dataSplits.length; i++) {
                for (int j = 0; j < dataSplits[i].numInstances(); j++) {
                    Instance currentInstance = dataSplits[i].instance(j);
                    splitsClassDist[i][(int) currentInstance.classValue()]++;
                }
            }

            for (int i = 0; i < dataSplits.length; i++) {

                for (int toBeClassifiedAs = 0; toBeClassifiedAs < data.classAttribute().numValues(); toBeClassifiedAs++) {

                    double cost = 0;

                    for (int j = 0; j < data.classAttribute().numValues(); j++) {

                        cost += m_CostMatrix.getElement(toBeClassifiedAs, j) * splitsClassDist[i][j];

                    }

                    splitsClassCosts[i][toBeClassifiedAs] = cost;

                }

            }

            for (int i = 0; i < dataSplits.length; i++) {
                double currentMinCost = Double.POSITIVE_INFINITY;

                for (int j = 0; j < numClasses; j++) {
                    if (splitsClassCosts[i][j] < currentMinCost) {
                        currentMinCost = splitsClassCosts[i][j];
                        splitsMinCostIndexes[i] = j;
                    }
                }
                splitsMinCostValueNames[i] = data.classAttribute().value(splitsMinCostIndexes[i]);
            }

            for (int i = 0; i < dataSplits.length; i++) {
                CSTree currentSubTree = new CSTree();
                currentSubTree.setConfidenceFactor(confidence);
                currentSubTree.setMinNumObj(minRecLeaf);
                currentSubTree.setCostMatrix(m_CostMatrix);
                currentSubTree.setCollapseTree(false); //as this is not used in siers implementation
                currentSubTree.setSubtreeRaising(false); //as this is not used in siers implementation
                if (dataSplits[i].numInstances() > 0) {
                    currentSubTree.buildClassifier(dataSplits[i]);
                    csTrees.add(currentSubTree);
                    numberLeaves += currentSubTree.measureNumLeaves();
                } else {
                    csTrees.add(null);
                    numberLeaves += 1;
                }

            }
        }

        /**
         * Returns string representation of forest.
         *
         * @return string representation of forest,
         */
        @Override
        public String toString() {
            String treeString = "";

            Attribute rootAttribute = rootSplit.getAttribute();
            double splitValue = rootSplit.getSplitPoint();

            if (rootAttribute.isNumeric()) {
                //treeString += "|   ";
                treeString += rootAttribute.name();
                treeString += " ";
                treeString += "<=";
                treeString += " ";
                treeString += Utils.roundDouble(splitValue, m_numDecimalPlaces);

                if (csTrees.get(1) == null) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[1];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    double totalCost = 0;
                    for (int j = 0; j < splitsClassDist[1].length; j++) {
                        treeString += classNames[j] + "," + splitsClassDist[1][j] + ";";

                        try {
                            totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[1], j)
                                    * splitsClassDist[1][j];
                        } catch (Exception ex) {
                            Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        if (j == splitsMinCostIndexes[1]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[1][j];

                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[1][splitsMinCostIndexes[1]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ") [";
                    treeString += totalCost;
                    treeString += "]";
                } else if (csTrees.get(1).measureNumLeaves() == 1) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[1];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    double totalCost = 0;
                    for (int j = 0; j < splitsClassDist[1].length; j++) {
                        treeString += classNames[j] + ":" + splitsClassDist[1][j] + ";";

                        try {
                            totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[1], j)
                                    * splitsClassDist[1][j];
                        } catch (Exception ex) {
                            Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        if (j == splitsMinCostIndexes[1]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[1][j];
                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[1][splitsMinCostIndexes[1]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ") [";
                    treeString += totalCost;
                    treeString += "]";
                } else {
                    treeString += "\n";
                    String csTreeString = csTrees.get(1).toString();
                    // remove the first 3 lines and the last 4 lines
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.replace("\n", "\n|   ");
                    csTreeString = "|   " + csTreeString;
                    // the last line will have "|   ", so remove the last line.
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    treeString += csTreeString;
                }

                treeString += "\n";
                treeString += rootAttribute.name();
                treeString += " ";
                treeString += ">";
                treeString += " ";
                treeString += Utils.roundDouble(splitValue, m_numDecimalPlaces);

                if (csTrees.get(0) == null) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[0];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    double totalCost = 0;
                    for (int j = 0; j < splitsClassDist[0].length; j++) {
                        treeString += classNames[j] + "," + splitsClassDist[0][j] + ";";

                        try {
                            totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[0], j)
                                    * splitsClassDist[0][j];
                        } catch (Exception ex) {
                            Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        if (j == splitsMinCostIndexes[0]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[0][j];
                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[0][splitsMinCostIndexes[0]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ") [";
                    treeString += totalCost;
                    treeString += "]";
                } else if (csTrees.get(0).measureNumLeaves() == 1) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[0];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    double totalCost = 0;
                    for (int j = 0; j < splitsClassDist[0].length; j++) {
                        treeString += classNames[j] + "," + splitsClassDist[0][j] + ";";

                        try {
                            totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[0], j)
                                    * splitsClassDist[0][j];
                        } catch (Exception ex) {
                            Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                        }

                        if (j == splitsMinCostIndexes[0]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[0][j];
                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[0][splitsMinCostIndexes[0]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ") [";
                    treeString += totalCost;
                    treeString += "]";
                } else {
                    treeString += "\n";
                    String csTreeString = csTrees.get(0).toString();
                    // remove the first 3 lines and the last 4 lines
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    csTreeString = csTreeString.replace("\n", "\n|   ");
                    csTreeString = "|   " + csTreeString;
                    csTreeString.replaceAll("\n", "\n|   ");
                    // the last line will have "|   ", so remove the last line.
                    csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                    treeString += csTreeString;
                }
            } else if (rootAttribute.isNominal()) {
                for (int i = 0; i < csTrees.size(); i++) {
                    if (i > 0) {
                        treeString += "\n";
                    }
                    treeString += "";
                    treeString += rootAttribute.name();
                    treeString += " ";
                    treeString += "=";
                    treeString += " ";
                    treeString += rootAttribute.value(i);

                    if (csTrees.get(i) == null) {
                        treeString += ": ";
                        treeString += splitsMinCostValueNames[i];
                        treeString += " {";
                        // count the remaining support
                        int remainingSupport = 0;
                        double totalCost = 0;
                        for (int j = 0; j < splitsClassDist[i].length; j++) {
                            treeString += classNames[j] + "," + splitsClassDist[i][j] + ";";

                            try {
                                totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[i], j)
                                        * splitsClassDist[i][j];
                            } catch (Exception ex) {
                                Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                            }

                            if (j == splitsMinCostIndexes[i]) {
                                continue;
                            }
                            remainingSupport += splitsClassDist[i][j];
                        }
                        treeString += "} (";
                        treeString += (splitsClassDist[i][splitsMinCostIndexes[i]] + remainingSupport);
                        treeString += "/";
                        treeString += remainingSupport;
                        treeString += ") [";
                        treeString += totalCost;
                        treeString += "]";
                    } else if (csTrees.get(i).measureNumLeaves() == 1) {
                        treeString += ": ";
                        treeString += splitsMinCostValueNames[i];
                        treeString += " {";
                        // count the remaining support
                        int remainingSupport = 0;
                        double totalCost = 0;
                        for (int j = 0; j < splitsClassDist[i].length; j++) {
                            treeString += classNames[j] + "," + splitsClassDist[i][j] + ";";

                            try {
                                totalCost += m_CostMatrix.getElement(splitsMinCostIndexes[i], j)
                                        * splitsClassDist[i][j];
                            } catch (Exception ex) {
                                Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
                            }

                            if (j == splitsMinCostIndexes[i]) {
                                continue;
                            }
                            remainingSupport += splitsClassDist[i][j];
                        }
                        treeString += "} (";
                        treeString += (splitsClassDist[i][splitsMinCostIndexes[i]] + remainingSupport);
                        treeString += "/";
                        treeString += remainingSupport;
                        treeString += ") [";
                        treeString += totalCost;
                        treeString += "]";
                    } else {
                        treeString += "\n";
                        String csTreeString = csTrees.get(i).toString();
                        // remove the first 3 lines and the last 4 lines
                        csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                        csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                        csTreeString = csTreeString.substring(csTreeString.indexOf('\n') + 1);
                        csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                        csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                        csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                        csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                        csTreeString = csTreeString.replace("\n", "\n|   ");
                        csTreeString = "|   " + csTreeString;
                        // the last line will have "|   ", so remove the last line.
                        csTreeString = csTreeString.substring(0, csTreeString.lastIndexOf('\n'));
                        treeString += csTreeString;
                    }
                }
            }

            return treeString;
        }

    }

    private class CSForestLevelTwoTree extends AbstractClassifier {

        private static final long serialVersionUID = 8018080734563784558L;

        private ArrayList<Integer> distribution;
        private int majorityIndex;
//        private int[] splitsMajorityIndexes;
        private int[] splitsMinCostIndexes;
        private int[][] splitsClassDist;
        private double[][] splitsClassCosts;
//        private String[] splitsMajorityValueNames;
        private String[] splitsMinCostValueNames;
        private String majorityValueName;
        private CSForestTree[] subTrees = null;
        private GoodAttribute rootSplit;
        private int numClasses;
        private int numberLeaves = 0;

        public CSForestLevelTwoTree(CSForestTree[] subTrees, GoodAttribute rootSplit) {
            this.subTrees = subTrees;
            this.rootSplit = rootSplit;
            for (int i = 0; i < subTrees.length; i++) {
                numberLeaves += subTrees[i].numberLeaves;
            }
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            numClasses = data.numClasses();
            distribution = new ArrayList<Integer>();
            Instances[] dataSplits = splitData(data, rootSplit);
            for (int i = 0; i < dataSplits.length; i++) {
                distribution.add(0);
            }

            for (int i = 0; i < dataSplits.length; i++) {
                distribution.set(i, dataSplits[i].numInstances());
            }

            splitsClassDist = new int[dataSplits.length][numClasses];
            splitsClassCosts = new double[dataSplits.length][numClasses];
            splitsMinCostIndexes = new int[dataSplits.length];
            splitsMinCostValueNames = new String[dataSplits.length];

            for (int i = 0; i < dataSplits.length; i++) {
                for (int j = 0; j < dataSplits[i].numInstances(); j++) {
                    Instance currentInstance = dataSplits[i].instance(j);
                    splitsClassDist[i][(int) currentInstance.classValue()]++;
                }
            }

            for (int i = 0; i < dataSplits.length; i++) {

                for (int toBeClassifiedAs = 0; toBeClassifiedAs < data.classAttribute().numValues(); toBeClassifiedAs++) {

                    double cost = 0;

                    for (int j = 0; j < data.classAttribute().numValues(); j++) {

                        cost += m_CostMatrix.getElement(toBeClassifiedAs, j) * splitsClassDist[i][j];

                    }

                    splitsClassCosts[i][toBeClassifiedAs] = cost;

                }

            }

            int largestValueSupport = -1;
            for (int i = 0; i < distribution.size(); i++) {
                if (distribution.get(i) > largestValueSupport) {
                    largestValueSupport = distribution.get(i);
                    majorityIndex = i;
                }
            }

            int biggest = -1;
            int myIndex = -1;
            for (int i = 0; i < numClasses; i++) {
                if (splitsClassDist[majorityIndex][i] > biggest) {
                    myIndex = i;
                    biggest = splitsClassDist[majorityIndex][i];
                }
            }

            majorityValueName = data.classAttribute().value(myIndex);
            //majorityValueName = data.classAttribute().value(majorityIndex);

            for (int i = 0; i < dataSplits.length; i++) {
                double currentMinCost = Double.POSITIVE_INFINITY;

                for (int j = 0; j < numClasses; j++) {
                    if (splitsClassCosts[i][j] < currentMinCost) {
                        currentMinCost = splitsClassCosts[i][j];
                        splitsMinCostIndexes[i] = j;
                    }
                }
                splitsMinCostValueNames[i] = data.classAttribute().value(splitsMinCostIndexes[i]);
            }
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws java.lang.Exception {
            Attribute rootAttribute = rootSplit.getAttribute();
            double splitValue = rootSplit.getSplitPoint();
            if (rootAttribute.isNumeric()) {
                if (instance.value(rootAttribute) > splitValue) {
                    return subTrees[0].distributionForInstance(instance);
                } else {
                    return subTrees[1].distributionForInstance(instance);
                }
            } else if (rootAttribute.isNominal()) {
                return subTrees[(int) instance.value(rootAttribute)].distributionForInstance(instance);
            } else {
                double[] returnValue = new double[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    returnValue[i] = 0;
                }
                return returnValue;
            }
        }

        @Override
        public String toString() {
            String treeString = "";

            Attribute rootAttribute = rootSplit.getAttribute();
            double splitValue = rootSplit.getSplitPoint();

            if (rootAttribute.isNumeric()) {
                //treeString += "|   ";
                treeString += rootAttribute.name();
                treeString += " ";
                treeString += "<=";
                treeString += " ";
                treeString += Utils.roundDouble(splitValue, m_numDecimalPlaces);

                if (subTrees[1].measureNumLeaves() == 1) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[1];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    for (int j = 0; j < splitsClassDist[1].length; j++) {
                        treeString += classNames[j] + "," + splitsClassDist[1][j] + ";";
                        if (j == splitsMinCostIndexes[1]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[1][j];
                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[1][splitsMinCostIndexes[1]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ")";
                } else {
                    treeString += "\n";
                    String subTreeString = subTrees[1].toString();
                    subTreeString = subTreeString.replace("\n", "\n|   ");
                    subTreeString = "|   " + subTreeString;
                    // the last line will have "|   ", so remove the last line.
                    //subTreeString = subTreeString.substring(0, subTreeString.lastIndexOf('\n'));
                    treeString += subTreeString;
                }

                treeString += "\n";
                treeString += rootAttribute.name();
                treeString += " ";
                treeString += ">";
                treeString += " ";
                treeString += Utils.roundDouble(splitValue, m_numDecimalPlaces);

                if (subTrees[0].measureNumLeaves() == 1) {
                    treeString += ": ";
                    treeString += splitsMinCostValueNames[0];
                    treeString += " {";
                    // count the remaining support
                    int remainingSupport = 0;
                    for (int j = 0; j < splitsClassDist[0].length; j++) {
                        treeString += classNames[j] + "," + splitsClassDist[0][j] + ";";
                        if (j == splitsMinCostIndexes[0]) {
                            continue;
                        }
                        remainingSupport += splitsClassDist[0][j];
                    }
                    treeString += "} (";
                    treeString += (splitsClassDist[0][splitsMinCostIndexes[0]] + remainingSupport);
                    treeString += "/";
                    treeString += remainingSupport;
                    treeString += ")";
                } else {
                    treeString += "\n";
                    String subTreeString = subTrees[0].toString();
                    subTreeString = subTreeString.replace("\n", "\n|   ");
                    subTreeString = "|   " + subTreeString;
                    subTreeString.replaceAll("\n", "\n|   ");
                    // the last line will have "|   ", so remove the last line.
                    //subTreeString = subTreeString.substring(0, subTreeString.lastIndexOf('\n'));
                    treeString += subTreeString;
                }
            } else if (rootAttribute.isNominal()) {
                for (int i = 0; i < subTrees.length; i++) {
                    if (i > 0) {
                        treeString += "\n";
                    }
                    treeString += "";
                    treeString += rootAttribute.name();
                    treeString += " ";
                    treeString += "=";
                    treeString += " ";
                    treeString += rootAttribute.value(i);

                    if (subTrees[i].measureNumLeaves() == 1) {
                        treeString += ": ";
                        treeString += splitsMinCostValueNames[i];
                        treeString += " {";
                        // count the remaining support
                        int remainingSupport = 0;
                        for (int j = 0; j < splitsClassDist[i].length; j++) {
                            treeString += classNames[j] + "," + splitsClassDist[i][j] + ";";
                            if (j == splitsMinCostIndexes[i]) {
                                continue;
                            }
                            remainingSupport += splitsClassDist[i][j];
                        }
                        treeString += "} (";
                        treeString += (splitsClassDist[i][splitsMinCostIndexes[i]] + remainingSupport);
                        treeString += "/";
                        treeString += remainingSupport;
                        treeString += ")";
                    } else {
                        treeString += "\n";
                        String subTreeString = subTrees[i].toString();
                        subTreeString = subTreeString.replace("\n", "\n|   ");
                        subTreeString = "|   " + subTreeString;
                        // the last line will have "|   ", so remove the last line.
                        //subTreeString = subTreeString.substring(0, subTreeString.lastIndexOf('\n'));
                        treeString += subTreeString;
                    }
                }
            }

            return treeString;
        }

    }

    /**
     * Splits a given set of instances into multiple instance based on a split
     * attribute. A split value is also used if the split attribute is
     * numerical.
     *
     * @param data the instances to split
     * @param splitPoint the object containing the split attribute and split
     * value. The split value is not used if the attribute is nominal.
     * @return an array of the data splits.
     */
    private Instances[] splitData(Instances data, GoodAttribute splitPoint) {
        Instances[] dataSplits = null;
        int numBags = -1;

        Attribute splitAttribute = splitPoint.getAttribute();
        double splitValue = splitPoint.getSplitPoint();
        if (splitAttribute.isNumeric()) {
            numBags = 2;
            dataSplits = new Instances[numBags];
            dataSplits[0] = new Instances(data, 0);
            dataSplits[1] = new Instances(data, 0);
            Enumeration eData = data.enumerateInstances();
            while (eData.hasMoreElements()) {
                Instance currentInstance = (Instance) eData.nextElement();
                double currentValue = currentInstance.value(splitAttribute);
                if (currentValue > splitValue) {
                    dataSplits[0].add(currentInstance);
                } else {
                    dataSplits[1].add(currentInstance);
                }
            }
        } else if (splitAttribute.isNominal()) {
            numBags = splitAttribute.numValues();

            dataSplits = new Instances[numBags];
            for (int i = 0; i < numBags; i++) {
                dataSplits[i] = new Instances(data, 0);
            }

            Enumeration eData = data.enumerateInstances();
            while (eData.hasMoreElements()) {
                Instance currentInstance = (Instance) eData.nextElement();
                double currentValue = currentInstance.value(splitAttribute);

                dataSplits[(int) currentValue].add(currentInstance);
            }
        }

        return dataSplits;
    }

    /**
     * Given an enumeration e, return a sorted enumeration from lowest to
     * highest
     *
     * @param e the enumeration to be sorted
     * @return a sorted enumeration
     */
    private Enumeration sortEnumeration(Enumeration e) {
        // Convert the Enumeration to a list, then sort the list.
        List<Double> values = Collections.list(e);
        Collections.sort(values);

        // Now convert the list back into an Enumeration object.
        Enumeration<Double> returnValue = new Vector(values).elements();
        return returnValue;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Michael J. Siers & Md Zahidul Islam");
        result.setValue(Field.YEAR, "2015");
        result.setValue(Field.TITLE, "Software defect prediction using a cost sensitive decision forest and voting, and a potential solution to the class imbalance problem");
        result.setValue(Field.JOURNAL, "Information Systems");
        result.setValue(Field.PUBLISHER, "Elsevier");
        result.setValue(Field.VOLUME, "51");
        result.setValue(Field.PAGES, "62-71");
        result.setValue(Field.URL, "http://dx.doi.org/10.1016/j.is.2015.02.006");

        return result;

    }

    /**
     * Turns the forest into a string to be printed.
     *
     * @return string representation of the built forest
     */
    @Override
    public String toString() {

        if (forest == null) {
            return "No Forest Built!";
        }

        String forestString = "";

        if (!correctlySetCostMatrix) {
            forestString += "Cost matrix incorrectly set (not enough rows/columns for classes):\n\n";
        }

        for (int i = 0; i < forest.size(); i++) {
            if (i != 0) {
                forestString += "\n";
            }
            forestString += "Tree " + (i + 1) + ": \n";
            forestString += forest.get(i);

            if (calculateTotalCost) {
                double treeCost = 0;
                //get the full costs for these trees
                String treeToCheck = forest.get(i).toString();

                Pattern pattern = Pattern.compile("\\[[0-9\\.]+\\]");
                Matcher matcher = pattern.matcher(treeToCheck);
                // check all occurance
                while (matcher.find()) {
                    treeCost += Double.parseDouble(matcher.group().substring(1, matcher.group().length() - 1));
                }
                forestString += "\nTree " + (i + 1) + " Cost: " + treeCost;
            }

            forestString += "\n";
        }

        //if the total cost should be calculated
        if (calculateTotalCost) {

            double costTotal = 0;

            try {

                //iterate over the dataset and grab the predictions
                for (int i = 0; i < dataset.size(); i++) {

                    double actual = dataset.get(i).classValue();
                    double prediction = classifyInstance(dataset.get(i));

                    costTotal += m_CostMatrix.getElement((int) prediction, (int) actual);

                }

            } catch (Exception ex) {
                Logger.getLogger(CSForest.class.getName()).log(Level.SEVERE, null, ex);
            }

            forestString += "\n\nTotal classification cost: " + costTotal;

        }

        return forestString;
    }

    /**
     * Sets the misclassification cost matrix.
     *
     * @param newCostMatrix the cost matrix
     */
    public void setCostMatrix(CostMatrix newCostMatrix) {

        m_CostMatrix = newCostMatrix;

    }

    /**
     * Getter for cost matrix
     *
     * @return m_CostMatrix
     */
    public CostMatrix getCostMatrix() {
        return this.m_CostMatrix;
    }

    /**
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String costMatrixTipText() {
        return "The cost matrix in Matlab single line format.\n"
                + "Expanded cost matrix takes form:\n"
                + "                   Actual class\n"
                + "                        | |\n"
                + "                        v v\n"
                + "                        a b\n"
                + "   class to be   =>  a  0 5\n"
                + " classified as   =>  b  1 1";
    }

    /**
     * Setter for minRecLeaf
     *
     * @param minRecLeaf value to set to
     */
    public void setMinRecLeaf(int minRecLeaf) {
        this.minRecLeaf = minRecLeaf;
    }

    /**
     * Getter for minimum leaf records
     *
     * @return minRecLeaf
     */
    public int getMinRecLeaf() {
        return this.minRecLeaf;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String minRecLeafTipText() {
        return "Minimum number of records required in each leaf.";
    }

    /**
     * Setter for numberTrees
     *
     * @param numberTrees value to set to
     */
    public void setNumberTrees(int numberTrees) {
        this.numberTrees = numberTrees;
    }

    /**
     * Getter for numberTrees
     *
     * @return number trees
     */
    public int getNumberTrees() {
        return this.numberTrees;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String numberTreesTipText() {
        return "Number of trees to build.";
    }

    /**
     * Setter for goodness
     *
     * @param costGoodness value to set to
     */
    public void setCostGoodness(float costGoodness) {
        this.costGoodness = costGoodness;
    }

    /**
     * Getter for goodness
     *
     * @return goodness value
     */
    public float getCostGoodness() {
        return this.costGoodness;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String costGoodnessTipText() {
        return "\"Good\" attributes must be within this threshold of distance"
                + "from the best available attribute.";
    }

    /**
     * Setter for separation
     *
     * @param separation value to set to
     */
    public void setSeparation(float separation) {
        this.separation = separation;
    }

    /**
     * Getter for separation
     *
     * @return separation
     */
    public float getSeparation() {
        return this.separation;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String separationTipText() {
        return "The required separation between two \"good\" split points.";
    }

    /**
     * Setter for calculateTotalCost
     *
     * @param calculateTotalCost value to set to
     */
    public void setCalculateTotalCost(boolean calculateTotalCost) {
        this.calculateTotalCost = calculateTotalCost;
    }

    /**
     * Getter for calculateTotalCost
     *
     * @return calculateTotalCost
     */
    public boolean getCalculateTotalCost() {
        return this.calculateTotalCost;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String calculateTotalCostTipText() {
        return "Whether or not to calculate the total cost of classification on the input dataset.";
    }

    /**
     * Getter for confidence
     *
     * @return Confidence factor for decision trees.
     */
    public float getConfidence() {
        return confidence;
    }

    /**
     * Setter for confidence
     *
     * @param confidence value to set to
     */
    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String confidenceTipText() {
        return "Confidence factor for decision trees.";
    }

    /**
     * Parses a given list of options.
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre>
     * -L &lt;minimum records in leaf&gt;
     *  Set minimum number of records for a leaf.
     *  (default 10)
     * </pre>
     *
     * <pre>
     * -N &lt;no. trees&gt;
     *  Set number of trees to build.
     *  (default 60)
     * </pre>
     *
     * <pre>
     * -G &lt;goodness threshold&gt;
     *  Set goodness threshold for attribute selection.
     *  (default 0.3)
     * </pre>
     *
     * <pre>
     * -S &lt;separation threshold&gt;
     *  Set separation threshold for split point selection.
     *  (default 0.3)
     * </pre>
     *
     * <pre>
     * -A 
     *  Whether to calculate the total classification cost of the training dataset.
     *  (default false)
     * </pre>
     *
     * <pre>
     * -C &lt;confidence factor&gt;
     *  Set confidence for pruning.
     *  (default 0.25)
     * </pre>
     *
     * <pre> -cost-matrix &lt;matrix&gt;
     *  The cost matrix in Matlab single line format.
     *  Expanded cost matrix takes form:
     *                    Actual class
     *                         | |
     *                         v v
     *                         a b
     *    class to be   =>  a  0 5
     *  classified as   =>  b  1 1
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        String sMinRecLeaf = Utils.getOption('L', options);
        if (sMinRecLeaf.length() != 0) {
            setMinRecLeaf(Integer.parseInt(sMinRecLeaf));
        } else {
            setMinRecLeaf(10);
        }

        String sNumberTrees = Utils.getOption('N', options);
        if (sNumberTrees.length() != 0) {
            setNumberTrees(Integer.parseInt(sNumberTrees));
        } else {
            setNumberTrees(60);
        }

        String sGoodness = Utils.getOption('G', options);
        if (sGoodness.length() != 0) {
            setCostGoodness(Float.parseFloat(sGoodness));
        } else {
            setCostGoodness(0.3f);
        }

        String sSeparation = Utils.getOption('S', options);
        if (sSeparation.length() != 0) {
            setSeparation(Float.parseFloat(sSeparation));
        } else {
            setSeparation(0.3f);
        }

        String sConfidence = Utils.getOption('C', options);
        if (sConfidence.length() != 0) {
            setConfidence(Float.parseFloat(sConfidence));
        } else {
            setConfidence(0.25f);
        }

        boolean sCostCalc = Utils.getFlag('A', options);
        setCalculateTotalCost(sCostCalc);

        String cost_matrix = Utils.getOption("cost-matrix", options);
        if (cost_matrix.length() != 0) {
            StringWriter writer = new StringWriter();
            CostMatrix.parseMatlab(cost_matrix).write(writer);
            setCostMatrix(new CostMatrix(new StringReader(writer.toString())));
        } else {
            cost_matrix = "[0 5; 1 1]";
            StringWriter writer = new StringWriter();
            CostMatrix.parseMatlab(cost_matrix).write(writer);
            setCostMatrix(new CostMatrix(new StringReader(writer.toString())));

        }

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return the current setting of the classifier
     */
    @Override
    public String[] getOptions() {

        Vector<String> result = new Vector<String>();

        result.add("-L");
        result.add("" + getMinRecLeaf());

        result.add("-N");
        result.add("" + getNumberTrees());

        result.add("-G");
        result.add("" + getCostGoodness());

        result.add("-S");
        result.add("" + getSeparation());

        result.add("-C");
        result.add("" + getConfidence());

        if (calculateTotalCost) {
            result.add("-A");
        }

        result.add("-cost-matrix");
        result.add(getCostMatrix().toMatlab());

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * Valid options are:
     * <p>
     *
     * -L &lt;minimum records in leaf&gt; <br>
     * Set minimum number of records for a leaf. (default 10)
     * <p>
     *
     * -N &lt;no. trees&gt; <br>
     * Set number of trees to build. (default 60)
     * <p>
     *
     * -G &lt;goodness threshold&gt; <br>
     * Set goodness threshold for attribute selection. (default 0.3)
     * <p>
     *
     * -S &lt;separation threshold&gt; <br>
     * Set separation threshold for split point selection. (default 0.3)
     * <p>
     *
     * -C &lt;confidence factor&gt; <br>
     * Set confidence for pruning. (default 0.25)
     * <p>
     * 
     * -A <br>
     * Whether to calculate the total classification cost of the training dataset. (default false)
     * <p>
     *
     * -cost-matrix &lt;matrix&gt; <br>
     * The cost matrix in Matlab single line format.
     * <p>
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(13);

        newVector.addElement(new Option("\tSet minimum number of records for a leaf.\n"
                + "\t(default 10)", "L", 1, "-L"));
        newVector.addElement(new Option("\tSet number of trees to build.\n"
                + "\t(default 60)", "N", 1, "-N"));
        newVector.addElement(new Option("\tSet goodness threshold for attribute selection.\n"
                + "\t(default 0.3)", "G", 1, "-G"));
        newVector.addElement(new Option("\tSet separation threshold for split point selection.\n"
                + "\t(default 0.3)", "S", 1, "-S"));
        newVector.addElement(new Option("\tSet confidence for pruning.\n"
                + "\t(default 0.25)", "C", 1, "-C"));
        newVector.addElement(new Option("\tWhether to calculate the total classification cost of the training dataset.\n"
                + "\t(default false)", "A", 0, "-A"));
        newVector.addElement(new Option("\tThe cost matrix in Matlab single line format.\n"
                + "\t(default [0 5; 1 1])", "cost-matrix", 1, "-cost-matrix"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

}
