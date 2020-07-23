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
 *    CSTree.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.classifiers.trees.j48;

import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Sourcable;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Implementation of the CSTree algorithm for use in CSForest. CSTree is a
 * cost-sensitive version of C4.5.
 *
 * @author modified by Michael Furner (mfurner@csu.edu.au) (originally by Eibe
 * Frank (eibe@cs.waikato.ac.nz))
 * @version $Revision: 11194 $
 */
public class CSTree extends AbstractClassifier implements OptionHandler, Drawable,
        Matchable, Sourcable, WeightedInstancesHandler, Summarizable,
        AdditionalMeasureProducer, TechnicalInformationHandler, PartitionGenerator {

    /**
     * for serialization
     */
    static final long serialVersionUID = -309284209348230L;

    /**
     * The decision tree
     */
    protected CostBasedClassifierTree m_root;

    /**
     * Unpruned tree?
     */
    protected boolean m_unpruned = false;

    /**
     * Collapse tree?
     */
    protected boolean m_collapseTree = true;

    /**
     * Confidence level
     */
    protected float m_CF = 0.25f;

    /**
     * Minimum number of instances
     */
    protected int m_minNumObj = 2;

    /**
     * Binary splits on nominal attributes?
     */
    protected boolean m_binarySplits = false;

    /**
     * Subtree raising to be performed?
     */
    protected boolean m_subtreeRaising = true;

    /**
     * Cleanup after the tree has been built.
     */
    protected boolean m_noCleanup = false;

    /**
     * Random number seed for reduced-error pruning.
     */
    protected int m_Seed = 1;

    /**
     * Do not relocate split point to actual data value
     */
    protected boolean m_doNotMakeSplitPointActualValue;

    /**
     * The cost matrix
     */
    protected CostMatrix m_CostMatrix = new CostMatrix(1);

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for generating a pruned or unpruned CSTree decision tree. For more "
                + "information, see\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Ling, C. X., Yang, Q., Wang, J., & Zhang, S.");
        result.setValue(Field.YEAR, "2004");
        result.setValue(Field.TITLE, "Decision trees with minimal costs");
        result.setValue(Field.JOURNAL, "Proceedings of the twenty-first international conference on Machine learning");
        result.setValue(Field.PUBLISHER, "ACM");
        result.setValue(Field.PAGES, "69");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result;

        result = new Capabilities(this);
        result.disableAll();
        // attributes
        result.enable(Capability.BINARY_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.NOMINAL_CLASS);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances the data to train the classifier with
     * @throws Exception if classifier can't be built successfully
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {

        getCapabilities().testWithFail(instances);

        /* if the cost matrix hasn't been properly defined, we use a hollow
        matrix with all non-diagonal values as 1 */
        if (m_CostMatrix.size() != instances.numClasses()) {
            m_CostMatrix = new CostMatrix(instances.numClasses());
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

        ModelSelection modSelection;

        modSelection = new CSTreeModelSelection(m_minNumObj, instances,
                m_doNotMakeSplitPointActualValue, m_CostMatrix);

        m_root = new CSTreePruneableClassifierTree(modSelection, !m_unpruned, m_CF,
                m_subtreeRaising, !m_noCleanup, m_collapseTree, m_CostMatrix);

        m_root.buildClassifier(instances);

        ((CSTreeModelSelection) modSelection).cleanup();

    }

    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification for the instance
     * @throws Exception if instance can't be classified successfully
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }

    /**
     * Returns class probabilities for an instance.
     *
     * @param instance the instance to calculate the class probabilities for
     * @return the class probabilities
     * @throws Exception if distribution can't be computed successfully
     */
    @Override
    public final double[] distributionForInstance(Instance instance)
            throws Exception {

        double[] tmp = m_root.distributionForInstance(instance);

        return invertCosts(tmp);
    }

    /**
     * Returns class costs for an instance.
     *
     * @param instance the instance to calculate the class costs for
     * @return the class costs
     * @throws Exception if otion can't be computed successfully
     */
    public double[] distributionCosts(Instance instance) throws Exception {
        double[] tmp = m_root.distributionForInstance(instance);
        return tmp;
    }

    /**
     * Returns the type of graph this classifier represents.
     *
     * @return Drawable.TREE
     */
    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    /**
     * Returns graph describing the tree.
     *
     * @return the graph describing the tree
     * @throws Exception if graph can't be computed
     */
    @Override
    public String graph() throws Exception {

        return m_root.graph();
    }

    /**
     * Returns tree in prefix order.
     *
     * @return the tree in prefix order
     * @throws Exception if something goes wrong
     */
    @Override
    public String prefix() throws Exception {

        return m_root.prefix();
    }

    /**
     * Returns tree as an if-then statement.
     *
     * @param className the name of the Java class
     * @return the tree as a Java if-then type statement
     * @throws Exception if something goes wrong
     */
    @Override
    public String toSource(String className) throws Exception {

        StringBuffer[] source = m_root.toSource(className);
        return "class " + className + " {\n\n"
                + "  public static double classify(Object[] i)\n"
                + "    throws Exception {\n\n" + "    double p = Double.NaN;\n"
                + source[0] // Assignment code
                + "    return p;\n" + "  }\n" + source[1] // Support code
                + "}\n";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * Valid options are:
     * <p>
     *
     * -U <br>
     * Use unpruned tree.
     * <p>
     *
     * -C confidence <br>
     * Set confidence threshold for pruning. (Default: 0.25)
     * <p>
     *
     * -M number <br>
     * Set minimum number of instances per leaf. (Default: 2)
     * <p>
     *
     * -R <br>
     * Use reduced error pruning. No subtree raising is performed.
     * <p>
     *
     * -N number <br>
     * Set number of folds for reduced error pruning. One fold is used as the
     * pruning set. (Default: 3)
     * <p>
     *
     * -B <br>
     * Use binary splits for nominal attributes.
     * <p>
     *
     * -S <br>
     * Don't perform subtree raising.
     * <p>
     *
     * -L <br>
     * Do not clean up after the tree has been built.
     *
     * <p>
     *
     * -Q <br>
     * The seed for reduced-error pruning.
     * <p>
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(13);

        newVector.addElement(new Option("\tUse unpruned tree.", "U", 0, "-U"));
        newVector.addElement(new Option("\tDo not collapse tree.", "O", 0, "-O"));
        newVector.addElement(new Option("\tSet confidence threshold for pruning.\n"
                + "\t(default 0.25)", "C", 1, "-C <pruning confidence>"));
        newVector.addElement(new Option(
                "\tSet minimum number of instances per leaf.\n" + "\t(default 2)", "M",
                1, "-M <minimum number of instances>"));
        newVector.addElement(new Option("\tUse reduced error pruning.", "R", 0,
                "-R"));
        newVector.addElement(new Option("\tSet number of folds for reduced error\n"
                + "\tpruning. One fold is used as pruning set.\n" + "\t(default 3)", "N",
                1, "-N <number of folds>"));
        newVector.addElement(new Option("\tUse binary splits only.", "B", 0, "-B"));
        newVector.addElement(new Option("\tDo not perform subtree raising.", "S", 0,
                "-S"));
        newVector.addElement(new Option(
                "\tDo not clean up after the tree has been built.", "L", 0, "-L"));
        newVector.addElement(new Option(
                "\tDo not use MDL correction for info gain on numeric attributes.", "J",
                0, "-J"));
        newVector.addElement(new Option(
                "\tSeed for random data shuffling (default 1).", "Q", 1, "-Q <seed>"));
        newVector.addElement(new Option("\tDo not make split point actual value.",
                "-doNotMakeSplitPointActualValue", 0, "-doNotMakeSplitPointActualValue"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre>
     * -U
     *  Use unpruned tree.
     * </pre>
     *
     * <pre>
     * -O
     *  Do not collapse tree.
     * </pre>
     *
     * <pre>
     * -C &lt;pruning confidence&gt;
     *  Set confidence threshold for pruning.
     *  (default 0.25)
     * </pre>
     *
     * <pre>
     * -M &lt;minimum number of instances&gt;
     *  Set minimum number of instances per leaf.
     *  (default 2)
     * </pre>
     *
     * <pre>
     * -R
     *  Use reduced error pruning.
     * </pre>
     *
     * <pre>
     * -N &lt;number of folds&gt;
     *  Set number of folds for reduced error
     *  pruning. One fold is used as pruning set.
     *  (default 3)
     * </pre>
     *
     * <pre>
     * -B
     *  Use binary splits only.
     * </pre>
     *
     * <pre>
     * -S
     *  Don't perform subtree raising.
     * </pre>
     *
     * <pre>
     * -L
     *  Do not clean up after the tree has been built.
     * </pre>
     *
     *
     * <pre>
     * -J
     *  Do not use MDL correction for info gain on numeric attributes.
     * </pre>
     *
     * <pre>
     * -Q &lt;seed&gt;
     *  Seed for random data shuffling (default 1).
     * </pre>
     *
     * <pre>
     * -doNotMakeSplitPointActualValue
     *  Do not make split point actual value.
     * </pre>
     *
     * <pre> -cost-matrix &lt;matrix&gt;
     *  The cost matrix in Matlab single line format.</pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        // Other options
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            m_minNumObj = Integer.parseInt(minNumString);
        } else {
            m_minNumObj = 2;
        }
        m_binarySplits = Utils.getFlag('B', options);

        // Pruning options
        m_unpruned = Utils.getFlag('U', options);
        m_collapseTree = !Utils.getFlag('O', options);
        m_subtreeRaising = !Utils.getFlag('S', options);
        m_noCleanup = Utils.getFlag('L', options);
        m_doNotMakeSplitPointActualValue = Utils.getFlag(
                "doNotMakeSplitPointActualValue", options);
        if ((m_unpruned) && (!m_subtreeRaising)) {
            throw new Exception(
                    "Subtree raising doesn't need to be unset for unpruned tree!");
        }
        String confidenceString = Utils.getOption('C', options);
        if (confidenceString.length() != 0) {
            if (m_unpruned) {
                throw new Exception(
                        "Doesn't make sense to change confidence for unpruned " + "tree!");
            } else {
                m_CF = (new Float(confidenceString)).floatValue();
                if ((m_CF <= 0) || (m_CF >= 1)) {
                    throw new Exception(
                            "Confidence has to be greater than zero and smaller " + "than one!");
                }
            }
        } else {
            m_CF = 0.25f;
        }

        String seedString = Utils.getOption('Q', options);
        if (seedString.length() != 0) {
            m_Seed = Integer.parseInt(seedString);
        } else {
            m_Seed = 1;
        }

        String cost_matrix = Utils.getOption("cost-matrix", options);
        if (cost_matrix.length() != 0) {
            StringWriter writer = new StringWriter();
            CostMatrix.parseMatlab(cost_matrix).write(writer);
            setCostMatrix(new CostMatrix(new StringReader(writer.toString())));
        }
        super.setOptions(options);

        Utils.checkForRemainingOptions(options);
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
        return "Sets the cost matrix explicitly.";
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        Vector<String> options = new Vector<String>();

        if (m_noCleanup) {
            options.add("-L");
        }
        if (!m_collapseTree) {
            options.add("-O");
        }
        if (m_unpruned) {
            options.add("-U");
        } else {
            if (!m_subtreeRaising) {
                options.add("-S");
            }
            options.add("-C");
            options.add("" + m_CF);

        }
        if (m_binarySplits) {
            options.add("-B");
        }
        options.add("-M");
        options.add("" + m_minNumObj);
        if (m_doNotMakeSplitPointActualValue) {
            options.add("-doNotMakeSplitPointActualValue");
        }

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String seedTipText() {
        return "The seed used for randomizing the data "
                + "when reduced-error pruning is used.";
    }

    /**
     * Get the value of Seed.
     *
     * @return Value of Seed.
     */
    public int getSeed() {

        return m_Seed;
    }

    /**
     * Set the value of Seed.
     *
     * @param newSeed Value to assign to Seed.
     */
    public void setSeed(int newSeed) {

        m_Seed = newSeed;
    }

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier
     */
    @Override
    public String toString() {

        if (m_root == null) {
            return "No classifier built";
        }
        if (m_unpruned) {
            return "CSTree unpruned tree\n------------------\n" + m_root.toString();
        } else {
            return "CSTree pruned tree\n------------------\n" + m_root.toString();
        }
    }

    /**
     * Returns a superconcise version of the model
     *
     * @return a summary of the model
     */
    @Override
    public String toSummaryString() {

        return "Number of leaves: " + m_root.numLeaves() + "\n"
                + "Size of the tree: " + m_root.numNodes() + "\n";
    }

    /**
     * Returns the size of the tree
     *
     * @return the size of the tree
     */
    public double measureTreeSize() {
        return m_root.numNodes();
    }

    /**
     * Returns the number of leaves
     *
     * @return the number of leaves
     */
    public double measureNumLeaves() {
        return m_root.numLeaves();
    }

    /**
     * Returns the number of rules (same as number of leaves)
     *
     * @return the number of rules
     */
    public double measureNumRules() {
        return m_root.numLeaves();
    }

    /**
     * Returns an enumeration of the additional measure names
     *
     * @return an enumeration of the measure names
     */
    @Override
    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector<String>(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }

    /**
     * Returns the value of the named measure
     *
     * @param additionalMeasureName the name of the measure to query for its
     * value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    @Override
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return measureNumRules();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return measureTreeSize();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return measureNumLeaves();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (j48)");
        }
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String unprunedTipText() {
        return "Whether pruning is performed.";
    }

    /**
     * Get the value of unpruned.
     *
     * @return Value of unpruned.
     */
    public boolean getUnpruned() {

        return m_unpruned;
    }

    /**
     * Set the value of unpruned. Turns reduced-error pruning off if set.
     *
     * @param v Value to assign to unpruned.
     */
    public void setUnpruned(boolean v) {

        m_unpruned = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String collapseTreeTipText() {
        return "Whether parts are removed that do not reduce training error.";
    }

    /**
     * Get the value of collapseTree.
     *
     * @return Value of collapseTree.
     */
    public boolean getCollapseTree() {

        return m_collapseTree;
    }

    /**
     * Set the value of collapseTree.
     *
     * @param v Value to assign to collapseTree.
     */
    public void setCollapseTree(boolean v) {

        m_collapseTree = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String confidenceFactorTipText() {
        return "The confidence factor used for pruning (smaller values incur "
                + "more pruning).";
    }

    /**
     * Get the value of CF.
     *
     * @return Value of CF.
     */
    public float getConfidenceFactor() {

        return m_CF;
    }

    /**
     * Set the value of CF.
     *
     * @param v Value to assign to CF.
     */
    public void setConfidenceFactor(float v) {

        m_CF = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String minNumObjTipText() {
        return "The minimum number of instances per leaf.";
    }

    /**
     * Get the value of minNumObj.
     *
     * @return Value of minNumObj.
     */
    public int getMinNumObj() {
        return m_minNumObj;
    }

    /**
     * Set the value of minNumObj.
     *
     * @param v Value to assign to minNumObj.
     */
    public void setMinNumObj(int v) {
        m_minNumObj = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String reducedErrorPruningTipText() {
        return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String numFoldsTipText() {
        return "Determines the amount of data used for reduced-error pruning. "
                + " One fold is used for pruning, the rest for growing the tree.";
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String binarySplitsTipText() {
        return "Whether to use binary splits on nominal attributes when "
                + "building the trees.";
    }

    /**
     * Get the value of binarySplits.
     *
     * @return Value of binarySplits.
     */
    public boolean getBinarySplits() {

        return m_binarySplits;
    }

    /**
     * Set the value of binarySplits.
     *
     * @param v Value to assign to binarySplits.
     */
    public void setBinarySplits(boolean v) {

        m_binarySplits = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String subtreeRaisingTipText() {
        return "Whether to consider the subtree raising operation when pruning.";
    }

    /**
     * Get the value of subtreeRaising.
     *
     * @return Value of subtreeRaising.
     */
    public boolean getSubtreeRaising() {

        return m_subtreeRaising;
    }

    /**
     * Set the value of subtreeRaising.
     *
     * @param v Value to assign to subtreeRaising.
     */
    public void setSubtreeRaising(boolean v) {

        m_subtreeRaising = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String saveInstanceDataTipText() {
        return "Whether to save the training data for visualization.";
    }

    /**
     * Check whether instance data is to be saved.
     *
     * @return true if instance data is saved
     */
    public boolean getSaveInstanceData() {

        return m_noCleanup;
    }

    /**
     * Set whether instance data is to be saved.
     *
     * @param v true if instance data is to be saved
     */
    public void setSaveInstanceData(boolean v) {

        m_noCleanup = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String doNotMakeSplitPointActualValueTipText() {
        return "If true, the split point is not relocated to an actual data value."
                + " This can yield substantial speed-ups for large datasets with numeric attributes.";
    }

    /**
     * Gets the value of doNotMakeSplitPointActualValue.
     *
     * @return the value
     */
    public boolean getDoNotMakeSplitPointActualValue() {
        return m_doNotMakeSplitPointActualValue;
    }

    /**
     * Sets the value of doNotMakeSplitPointActualValue.
     *
     * @param m_doNotMakeSplitPointActualValue the value to set
     */
    public void setDoNotMakeSplitPointActualValue(
            boolean m_doNotMakeSplitPointActualValue) {
        this.m_doNotMakeSplitPointActualValue = m_doNotMakeSplitPointActualValue;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11194 $");
    }

    /**
     * Builds the classifier to generate a partition.
     *
     * @param data
     * @throws java.lang.Exception
     */
    @Override
    public void generatePartition(Instances data) throws Exception {

        buildClassifier(data);
    }

    /**
     * Computes an array that indicates node membership.
     *
     * @param inst
     */
    @Override
    public double[] getMembershipValues(Instance inst) throws Exception {

        return m_root.getMembershipValues(inst);
    }

    /**
     * Returns the number of elements in the partition.
     */
    @Override
    public int numElements() throws Exception {

        return m_root.numNodes();
    }

    /**
     * Inverts an array of per-class costs into the probability distribution
     * expected by Weka for classification. Handles situations with multiple
     * classes with zero cost.
     *
     * @param input the array of costs to invert
     * @return inverted cost array
     */
    public static double[] invertCosts(double[] input) {

        double[] returnValue = input.clone();

        //invert the costs (make smaller costs have bigger value and larger costs
        //have smaller value)
        double total = 1;
        ArrayList<Integer> zeroes = new ArrayList<>();
        for (int i = 0; i < returnValue.length; i++) {
            if (returnValue[i] == 0) {
                returnValue[i] = Double.MAX_VALUE;
                zeroes.add(i);
            } else {
                returnValue[i] = total / returnValue[i];
            }
        }

        if (zeroes.size() > 0) { //handle situations with one or more 0 cost options
            for (int i = 0; i < returnValue.length; i++) {
                if (zeroes.contains(i)) {
                    returnValue[i] = 1;
                } else {
                    returnValue[i] = 0;
                }
            }
        }

        Utils.normalize(returnValue);

        return returnValue;

    }

}
