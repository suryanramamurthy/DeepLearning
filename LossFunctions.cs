using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning
{
    public abstract class Loss
    {
        // calculate the cumulative loss for the given actual and predicted values
        public abstract float getLoss(float[,] actual, float[,] predicted, bool batchAverage);
        // return the gradient for the Loss w.r.t. the predicted value
        public abstract float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage);
        // This will be added to the values in the denominator to ensure that they are not zero.
        protected float epsilon = 1e-15F;
        protected int batch, n;

        protected bool checkDimensions(float[,] actual, float[,] predicted)
        {
            // check if the length for each dimension matches between actual and predicted
            for (int i = 0; i < actual.Rank; i++)
                if (actual.GetLength(i) != predicted.GetLength(i))
                    return false;
            return true;
        }

        protected void getDimensions(float[,] actual)
        {
            this.batch = actual.GetLength(0);
            this.n = actual.GetLength(1);
        }
    }
    /*
    * First dimension represents the batch axis. The second dimension represents the input/output
    * feature dimension. The loss value will be summed across both dimensions.
    *
    * The gradient value will have ths same dimension as the feature axis. The gradient will be 
    * summed across the batch dimension (1st dimesnion)
    * 
    * If batchAverage is set to True, then the gradient will be averaged across the batch axis, the
    * sum will be returned for the batch axis.
    *
    * For Loss functions, the dimensions of the input and output features has to be the same.
    *
    * Mean Squared Error is defined as 
    * Loss = (1/n)Sigma[(actual<i> - predicted<i>)^2]
    * Gradient[i] = -2*(actual[i] - predicted[i])
    */

    public class MSE : Loss
    {
        // First dimension represents the batch axis. The second dimension represents the input/output
        // feature dimension. The loss value will be summed across both dimensions.

        // The gradient value will have ths same dimension as the feature axis. The gradient will be 
        // summed across the batch dimension (1st dimesnion)

        // For Loss functions, the dimensions of the input and output features has to be the same.
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] += -2.0F * (actual[j, i] - predicted[j, i]);
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float sumSquared = 0.0F;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    sumSquared += (float)Math.Pow((double)(actual[j, i] - predicted[j, i]), 2.0);
            sumSquared /= n;
            if (batchAverage) sumSquared /= batch;
            return sumSquared;
        }
    }

    public class MSLE : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = (float)(-2.0F * ((Math.Log(actual[j, i] + 1) - Math.Log(predicted[j, i] + 1)) / (predicted[j, i] + 1 + epsilon)));
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += (float)Math.Pow(Math.Log(actual[j, i] + 1) - Math.Log(predicted[j, i] + 1), 2);
            loss /= n;
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class L2 : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = -2.0F * (actual[j, i] - predicted[j, i]);
            if (batchAverage)
                for (int i = 0; i < batch; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float sumSquared = 0.0F;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    sumSquared += (float)Math.Pow((double)(actual[j, i] - predicted[j, i]), 2.0);
            if (batchAverage) sumSquared /= batch;
            return sumSquared;
        }
    }

    public class MAE : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                {
                    if (predicted[j, i] > actual[j, i]) gradient[i] = 1.0F;
                    if (predicted[j, i] < actual[j, i]) gradient[i] = -1.0F;
                    if (predicted[j, i] == actual[j, i]) gradient[i] = 0;
                }
            if (batchAverage)
                for (int i = 0; i < batch; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += Math.Abs(actual[j, i] - predicted[j, i]);
            loss /= n;
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class L1 : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                {
                    if (predicted[j, i] > actual[j, i]) gradient[i] = 1.0F;
                    if (predicted[j, i] < actual[j, i]) gradient[i] = -1.0F;
                    if (predicted[j, i] == actual[j, i]) gradient[i] = 0;
                }
            if (batchAverage)
                for (int i = 0; i < batch; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += Math.Abs(actual[j, i] - predicted[j, i]);
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class KLDivergenceLoss : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = -actual[j, i] / (predicted[j, i] + epsilon);
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += (float)(actual[j, i] * (Math.Log(actual[j, i]) - Math.Log(predicted[j, i])));
            loss /= n;
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class MultiClassCrossEntropyLoss : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = ((1 - actual[j, i]) / (1 - predicted[j, i])) - (actual[j, i] / (predicted[j, i] + epsilon));
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += (float)((actual[j, i] * Math.Log(predicted[j, i])) + ((1 - actual[j, i]) * Math.Log(1 - predicted[j, i])));
            loss /= -n;
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class NegativeLogLikelihood : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = -1.0F / (predicted[j, i] + epsilon);
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += (float)Math.Log(predicted[j, i]);
            loss /= -n;
            if (batchAverage) loss /= batch;
            return loss;
        }
    }

    public class CrossEntropyLoss : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = -(actual[j, i] / (predicted[j, i] + epsilon));
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += (float)(actual[j, i] * Math.Log(predicted[j, i]));
            if (batchAverage) loss /= batch;
            return -loss;
        }
    }

    public class Poisson : Loss
    {
        public override float[] dLdyp(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float[] gradient = new float[n];
            gradient.Initialize(); // Initialize all values to 0
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    gradient[i] = 1.0F - (actual[j,i] / (predicted[j,i] + epsilon));
            if (batchAverage)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] /= batch;
            return gradient;
        }

        public override float getLoss(float[,] actual, float[,] predicted, bool batchAverage)
        {
            if (checkDimensions(actual, predicted) == false)
                throw new ArgumentException("The dimensions of actual and predicted does not match");
            getDimensions(actual);
            float loss = 0;
            for (int j = 0; j < batch; j++)
                for (int i = 0; i < n; i++)
                    loss += predicted[j, i] - (actual[j, i] * (float)Math.Log(predicted[j, i]));
            if (batchAverage) loss /= batch;
            return -loss;
        }
    }

}
