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
 *    CSTreeModelSelection.java
 *
 */
package weka.classifiers.trees.j48;

import java.util.Enumeration;
import weka.classifiers.CostMatrix;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Class for selecting a C4.5-type split for a given dataset.
 *
 * @author modified by Michael Furner (mfurner@csu.edu.au) (originally by Eibe Frank (eibe@cs.waikato.ac.nz))
 * @version $Revision: 10531 $
 */
public class CSTreeModelSelection extends ModelSelection {

    /**
     * for serialization
     */
    private static final long serialVersionUID = 3372204862440821989L;

    /**
     * Minimum number of objects in interval.
     */
    protected final int m_minNoObj;


    /**
     * All the training data
     */
    protected Instances m_allData; //

    /**
     * Do not relocate split point to actual data value
     */
    protected final boolean m_doNotMakeSplitPointActualValue;

    /**
     * The Cost Matrix
     */
    protected CostMatrix m_CostMatrix;

    /**
     * Initializes the split selection method with the given parameters.
     *
     * @param minNoObj minimum number of instances that have to occur in at
     * least two subsets induced by split
     * @param allData FULL training dataset (necessary for selection of split
     * points).
     * @param useMDLcorrection whether to use MDL adjustement when finding
     * splits on numeric attributes
     * @param doNotMakeSplitPointActualValue if true, split point is not
     * relocated by scanning the entire dataset for the closest data value
     * @param costMatrix the cost matrix for the tree building process
     */
    public CSTreeModelSelection(int minNoObj, Instances allData,
            boolean doNotMakeSplitPointActualValue, CostMatrix costMatrix) {
        m_minNoObj = minNoObj;
        m_allData = allData;
        m_doNotMakeSplitPointActualValue = doNotMakeSplitPointActualValue;
        m_CostMatrix = costMatrix;

    }

    /**
     * Sets reference to training data to null.
     */
    public void cleanup() {

        m_allData = null;
    }
    
    /**
     * Selects CSTree-type split for the given dataset.
     *
     * @param data
     * @return the split model
     * @throws java.lang.Exception - if there is a porblem with the cost matrix
     */
    @Override
    public final ClassifierSplitModel selectModel(Instances data) throws Exception {
        double minResult;
        CSTreeSplit[] currentModel;
        CSTreeSplit bestModel = null;
        NoSplit noSplitModel = null;
        double averageInfoGain = 0;
        int validModels = 0;
        boolean multiVal = true;
        Distribution checkDistribution;
        Attribute attribute;
        double sumOfWeights;
        int i;
        double totalExpectedCost = 0;

        //calculate full expected cost
        double[] fullCostArray = new double[data.classAttribute().numValues()];
        for (int x = 0; x < data.classAttribute().numValues(); x++) {

            //for each of the possible class values we need to calculate 
            //classification and misclassification costs
            String thisClassValue = data.classAttribute().value(x);

            double runningSum = 0;

            for (int y = 0; y < data.classAttribute().numValues(); y++) {

                RemoveWithValues rmv = new RemoveWithValues();
                int classIdx = data.classIndex() + 1;
                rmv.setAttributeIndex("" + classIdx);
                int[] indicesArr = {y};
                rmv.setNominalIndicesArr(indicesArr);
                rmv.setInvertSelection(true);
                rmv.setInputFormat(new Instances(data));

                Instances temp = Filter.useFilter(new Instances(data), rmv); //temp dataset with only this class j
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

        // Check if all Instances belong to one class or if not
        // enough Instances to split.
        checkDistribution = new Distribution(data);
        noSplitModel = new NoSplit(checkDistribution);
        if (Utils.sm(checkDistribution.total(), 2 * m_minNoObj)
                || Utils.eq(checkDistribution.total(),
                        checkDistribution.perClass(checkDistribution.maxClass()))) {
            return noSplitModel;
        }

        // Check if all attributes are nominal and have a
        // lot of values.
        if (m_allData != null) {
            Enumeration<Attribute> enu = data.enumerateAttributes();
            while (enu.hasMoreElements()) {
                attribute = enu.nextElement();
                if ((attribute.isNumeric())
                        || (Utils.sm(attribute.numValues(),
                                (0.3 * m_allData.numInstances())))) {
                    multiVal = false;
                    break;
                }
            }
        }

        currentModel = new CSTreeSplit[data.numAttributes()];
        sumOfWeights = data.sumOfWeights();

        // For each attribute.
        for (i = 0; i < data.numAttributes(); i++) {

            // Apart from class attribute.
            if (i != (data).classIndex()) {

                // Get models for current attribute.
                currentModel[i] = new CSTreeSplit(i, m_minNoObj, sumOfWeights,
                        m_CostMatrix, totalExpectedCost);
                currentModel[i].buildClassifier(data);

                // Check if useful split for current attribute
                // exists and check for enumerated attributes with
                // a lot of values.
                if (currentModel[i].checkModel()) {
                    if (m_allData != null) {
                        if ((data.attribute(i).isNumeric()) || (multiVal || Utils.sm(data.attribute(i).numValues(),
                                (0.3 * m_allData.numInstances())))) {
                            validModels++;
                        }
                    } else {

                        //ADD COST BASED STUFF HERE
                        validModels++;
                    }
                }
            } else {
                currentModel[i] = null;
            }
        }

        // Check if any useful split was found.
        if (validModels == 0) {
            return noSplitModel;
        }

        // Find "best" attribute to split on.
        minResult = Double.MAX_VALUE;
        int bestModelIndex= -1;
        for (i = 0; i < data.numAttributes(); i++) {
            if ((i != (data).classIndex()) && (currentModel[i].checkModel())) {

                if (Utils.sm(currentModel[i].cost(), minResult)) {
                    bestModel = currentModel[i];
                    minResult = currentModel[i].cost();
                    bestModelIndex = i;
                }

            }
        }

        // Add all Instances with unknown values for the corresponding
        // attribute to the distribution for the model, so that
        // the complete distribution is stored with the model.
        bestModel.distribution().addInstWithUnknown(data, bestModel.attIndex());

        // Set the split point analogue to C45 if attribute numeric.
        if ((m_allData != null) && (!m_doNotMakeSplitPointActualValue)) {
            bestModel.setSplitPoint(m_allData);
        }
        return bestModel;
    }

    /**
     * Selects CSTree-type split for the given dataset.
     *
     * @param train training dataset
     * @param test testing dataset
     * @return the split model
     * @throws java.lang.Exception - if there is a problem with the cost matrix
     */
    @Override
    public final ClassifierSplitModel selectModel(Instances train, Instances test) throws Exception {

        return selectModel(train);
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10531 $");
    }
}
