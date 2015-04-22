package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;

import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
    private static final Logger logger =
        LoggerFactory.getLogger(MatrixFactCore.class);

    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
            DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
        // TODO
    	int rowId = r.userId;
    	int colId = r.prodId;
    	
    	// get and cache the user factor vector.
    	DoubleRow rowCache = new DenseDoubleRow(K+1);
    	DoubleRow userRow = LTable.get(rowId);
    	rowCache.reset(userRow);
    	    	
    	//get and cache the movie factor vector.
    	DoubleRow colCache = new DenseDoubleRow(K+1);
    	DoubleRow movieRow = RTable.get(colId);
    	colCache.reset(movieRow);
    		
    	double ni = rowCache.getUnlocked(K);
    	double mj = colCache.getUnlocked(K);
    	
    	double dotP = 0.0;
    	for(int i = 0 ; i < K; i++)
    	{
    		dotP += rowCache.getUnlocked(i)*colCache.getUnlocked(i);
    	}
    	
    	double eij = r.rating - dotP;
    	
    	// update user column
    	DoubleRowUpdate update1 = new DenseDoubleRowUpdate(K+1);
    	DoubleRowUpdate update2 = new DenseDoubleRowUpdate(K+1);
    	
    	
    	for(int i = 0; i < K ; i++)
    	{
    		update1.setUpdate(i,2*learningRate*(colCache.getUnlocked(i)*eij - (lambda/ni)*rowCache.getUnlocked(i)));
    		update2.setUpdate(i,2*learningRate*(rowCache.getUnlocked(i)*eij - (lambda/mj)*colCache.getUnlocked(i)));
    	}
    	
    	LTable.batchInc(rowId,update1);
    	RTable.batchInc(colId,update2);
    	
    }

    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
            int elemBegin, int elemEnd, DoubleTable LTable,
            DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
            int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {
        // TODO
        double sqLoss = 0;
        double totalLoss = 0;
        double l2Loss = 0.0;
        
        for(int i = LRowBegin; i < LRowEnd; i++)
        {
        	DoubleRow rowCache = new DenseDoubleRow(K+1);
        	DoubleRow userRow = LTable.get(i);
        	rowCache.reset(userRow);
        	
        	for(int j = 0 ; j < K; j++)
        	{
        		l2Loss += rowCache.getUnlocked(j)*rowCache.getUnlocked(j);
        	}
        }
        
        for(int i = RRowBegin; i < RRowEnd; i++)
        {
        	DoubleRow rowCache = new DenseDoubleRow(K+1);
        	DoubleRow prodRow = RTable.get(i);
        	rowCache.reset(prodRow);
        	
        	for(int j = 0 ; j < K; j++)
        	{
        		l2Loss += rowCache.getUnlocked(j)*rowCache.getUnlocked(j);
        	}
        }
        
        l2Loss *= lambda;
        
        
        for(int i = elemBegin ;  i < elemEnd; i++)
        {
        	Rating rating = ratings.get(i);
        	
        	DoubleRow rowCache = new DenseDoubleRow(K+1);
        	DoubleRow userRow = LTable.get(rating.userId);
        	rowCache.reset(userRow);
        	
        	DoubleRow colCache = new DenseDoubleRow(K+1);
        	DoubleRow prodRow = RTable.get(rating.prodId);
        	colCache.reset(prodRow);
        	
        	double sum = 0.0;
        	for(int j = 0 ; j < K ; j++	)
        	{
        		sum += colCache.getUnlocked(j)*rowCache.getUnlocked(j);
        	}
        	
        	sqLoss += (rating.rating - sum)*(rating.rating - sum);
        }
        
        totalLoss = sqLoss + l2Loss;
        		
        
        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
