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
 *    CSTreeCostSplitCrit.java
 *
 */
package weka.classifiers.trees.j48;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.CostMatrix;
import weka.classifiers.trees.CSForest;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for computing the gain ratio for a given distribution.
 *
 * @author modified by Michael Furner (mfurner@csu.edu.au) (originally by Eibe Frank (eibe@cs.waikato.ac.nz))
 * @version $Revision: 10169 $
 */
public final class CSTreeCostSplitCrit extends SplitCriterion {

    /**
     * for serialization
     */
    private static final long serialVersionUID = -433336694718670930L;

    /**
     * The cost matrix
     */
    protected CostMatrix m_CostMatrix;

    /**
     * Make the split criteria with a cost matrix
     *
     * @param cm - the cost matrix
     */
    public CSTreeCostSplitCrit(CostMatrix cm) {
        m_CostMatrix = cm;
    }

    /**
     * This method is an implementation of the cost criterion for the given
     * distribution.
     *
     * @param bags - the distribution in the split
     * @param classAttr - not used but required
     * @return the cost of the split
     * @throws java.lang.Exception - if there is a problem with the cost matrix
     */
    public final double splitCritValue(Distribution bags, Attribute classAttr) throws Exception {

        double[] costPerBag = new double[bags.actualNumBags()];

        for (int i = 0; i < bags.actualNumBags(); i++) {
            
            //dont count the cost of empty bags
            if(bags.perBag(i) == 0)
               continue;

            double[] costPerClassArray = new double[bags.numClasses()];

            for (int classToBeClassifiedAs = 0; classToBeClassifiedAs < bags.numClasses(); classToBeClassifiedAs++) {
                double runningSum = 0;

                for (int actualClassNumbersIdx = 0; actualClassNumbersIdx < bags.numClasses(); actualClassNumbersIdx++) {

                    double c_ij = m_CostMatrix.getElement(classToBeClassifiedAs, actualClassNumbersIdx) * bags.perClassPerBag(i, actualClassNumbersIdx);
                    runningSum += c_ij;

                }

                costPerClassArray[classToBeClassifiedAs] = runningSum; //check this

            }

            double numerator = 1;
            double denominator = 0;
            for (int k = 0; k < costPerClassArray.length; k++) {
                numerator *= costPerClassArray[k];
                denominator += costPerClassArray[k];
            }
            costPerBag[i] = 2 * numerator / denominator;

        }

        double costForSplit = 0;
        for (int i = 0; i < costPerBag.length; i++) {
            costForSplit += costPerBag[i];
        }
        return costForSplit;

    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10169 $");
    }
}
