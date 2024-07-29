import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NeuralNetwork2 {
    Matrix[] weights;
    Matrix[] biases;
    Matrix[] velocities;
    double learning_rate;
    double momentum_rate;
    int numLayers;

    public NeuralNetwork2(int inputSize, int[] hiddenSizes, int outputSize, double learning_rate, double momentum_rate) {
        this.learning_rate = learning_rate;
        this.momentum_rate = momentum_rate;
        this.numLayers = hiddenSizes.length + 2; // input layer + hidden layers + output layer

        weights = new Matrix[numLayers - 1];
        biases = new Matrix[numLayers - 1];
        velocities = new Matrix[numLayers - 1];

        // Initialize weights and biases
        int[] layerSizes = new int[numLayers];
        layerSizes[0] = inputSize;
        System.arraycopy(hiddenSizes, 0, layerSizes, 1, hiddenSizes.length);
        layerSizes[numLayers - 1] = outputSize;

        for (int i = 0; i < numLayers - 1; i++) { // numLayers = 3
            weights[i] = new Matrix(layerSizes[i + 1], layerSizes[i]);
            biases[i] = new Matrix(layerSizes[i + 1], 1, true);
            velocities[i] = new Matrix(layerSizes[i + 1], layerSizes[i], false);
        }
    }

    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);
        Matrix layerOutput = input;

        for (int i = 0; i < numLayers - 1; i++) {
            layerOutput = Matrix.multiply(weights[i], layerOutput);
            layerOutput.add(biases[i]);
            layerOutput.sigmoid();
        }
        return layerOutput.toArray();
    }

    public void train(double[] X, double[] Y) {
        Matrix input = Matrix.fromArray(X);
        Matrix[] layerOutputs = new Matrix[numLayers];
        layerOutputs[0] = input;

        // Forward pass
        for (int i = 0; i < numLayers - 1; i++) {
            layerOutputs[i + 1] = Matrix.multiply(weights[i], layerOutputs[i]);
            layerOutputs[i + 1].add(biases[i]);
            layerOutputs[i + 1].sigmoid();
        }

        Matrix target = Matrix.fromArray(Y);
        Matrix error = Matrix.subtract(target, layerOutputs[numLayers - 1]);

        // Backward pass
        Matrix[] gradients = new Matrix[numLayers - 1];
        gradients[numLayers - 2] = layerOutputs[numLayers - 1].dsigmoid();
        gradients[numLayers - 2].multiply(error);
        gradients[numLayers - 2].multiply(learning_rate);

        for (int i = numLayers - 3; i >= 0; i--) {
            Matrix who_T = Matrix.transpose(weights[i + 1]);
            Matrix hiddenErrors = Matrix.multiply(who_T, error);
            gradients[i] = layerOutputs[i + 1].dsigmoid();
            gradients[i].multiply(hiddenErrors);
            gradients[i].multiply(learning_rate);
            error = hiddenErrors;
        }

        // Update weights and biases
        for (int i = numLayers - 2; i >= 0; i--) {
            Matrix delta = Matrix.multiply(gradients[i], Matrix.transpose(layerOutputs[i]));
            velocities[i].multiply(momentum_rate);
//            delta.multiply(1 - momentum_rate);
            velocities[i].add(delta);
            weights[i].add(velocities[i]);
//            biases[i].add(gradients[i]);
        }
    }

    public void fit(double[][] data, double[][] desiredOutput, int epochs, int k) {
        int numSamples = data.length;
        int foldSize = numSamples / k;

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);

        double totalValidationLoss = 0.0;
        int patience = 10;
        double bestValidationLoss = Double.MAX_VALUE;

        for (int i = 0; i < k; i++) {
            int validationStart = i * foldSize;
            int validationEnd = validationStart + foldSize;

            double[][] trainData = new double[numSamples - foldSize][];
            double[][] trainDesiredOutput = new double[numSamples - foldSize][];
            double[][] validationData = new double[foldSize][];
            double[][] validationDesiredOutput = new double[foldSize][];

            int trainIndex = 0;
            int validationIndex = 0;

            for (int j = 0; j < numSamples; j++) {
                if (j >= validationStart && j < validationEnd) {
                    validationData[validationIndex] = data[indices.get(j)];
                    validationDesiredOutput[validationIndex] = desiredOutput[indices.get(j)];
                    validationIndex++;
                } else {
                    trainData[trainIndex] = data[indices.get(j)];
                    trainDesiredOutput[trainIndex] = desiredOutput[indices.get(j)];
                    trainIndex++;
                }
            }

            int patienceCounter = 0;
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int j = 0; j < trainData.length; j++) {
                    this.train(trainData[j], trainDesiredOutput[j]);
                }

                double validationLoss = computeLoss(validationData, validationDesiredOutput);
                System.out.println("Fold " + (i + 1) + ", Epoch " + (epoch + 1) + ", Validation Loss (MSE): " + validationLoss);

                if (validationLoss < bestValidationLoss) {
                    bestValidationLoss = validationLoss;
                    patienceCounter = 0; // Reset patience counter
                } else {
                    patienceCounter++;
                }

                if (patienceCounter >= patience) {
                    System.out.println("Early stopping on fold " + (i + 1) + " at epoch " + (epoch + 1));
                    break;
                }
            }

            totalValidationLoss += bestValidationLoss;
        }

        double averageValidationLoss = totalValidationLoss / k;
        System.out.println("Average Validation Loss (MSE): " + averageValidationLoss);
    }

    private double computeLoss(double[][] data, double[][] desiredOutput) {
        double totalLoss = 0.0;
        for (int i = 0; i < data.length; i++) {
            List<Double> prediction = predict(data[i]);
            double[] target = desiredOutput[i];
            for (int j = 0; j < target.length; j++) {
                double error = target[j] - prediction.get(j);
                totalLoss += error * error; // sum square error
            }
        }
        return totalLoss / data.length; // sum square average error
    }

    public double[] normalize(double[] data, double min, double max) {
        double[] normalizedData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normalizedData[i] = (data[i] - min) / (max - min);
        }
        return normalizedData;
    }

    public double[] denormalize(double[] normalizedData, double min, double max) {
        double[] denormalizedData = new double[normalizedData.length];
        for (int i = 0; i < normalizedData.length; i++) {
            denormalizedData[i] = normalizedData[i] * (max - min) + min;
        }
        return denormalizedData;
    }

    public double[] classifySample(double[] outputArray) {
        if (outputArray[0] > outputArray[1]) {
            return new double[]{1, 0}; // Class 1 0
        } else {
            return new double[]{0, 1}; // Class 0 1
        }
    }
}
