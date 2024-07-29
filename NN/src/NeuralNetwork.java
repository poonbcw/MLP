import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork {
    Matrix weights_ih, weights_ho, wbias_h, wbias_o, weights_ih_velocity, weights_ho_velocity;
    double learning_rate;
    double momentum_rate;

    public NeuralNetwork(int i, int h, int o, double learning_rate, double momentum_rate) {
        weights_ih = new Matrix(h, i);
        weights_ho = new Matrix(o, h);
        wbias_h = new Matrix(h, 1, true);
        wbias_o = new Matrix(o, 1, true);
        weights_ih_velocity = new Matrix(h, i, false);
        weights_ho_velocity = new Matrix(o, h, false);
        this.learning_rate = learning_rate;
        this.momentum_rate = momentum_rate;
    }

    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(wbias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho, hidden);
        output.add(wbias_o);
        output.sigmoid();

        return output.toArray();
    }

    public void train(double[] X, double[] Y) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(wbias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho, hidden);
        output.add(wbias_o);
        output.sigmoid();

        Matrix target = Matrix.fromArray(Y);
        Matrix error = Matrix.subtract(target, output);

        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(learning_rate);

        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta = Matrix.multiply(gradient, hidden_T);

        weights_ho_velocity.multiply(momentum_rate);
        weights_ho_velocity.add(who_delta);
        weights_ho.add(weights_ho_velocity);

        Matrix who_T = Matrix.transpose(weights_ho);
        Matrix hidden_errors = Matrix.multiply(who_T, error);

        Matrix h_gradient = hidden.dsigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(learning_rate);

        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

        weights_ih_velocity.multiply(momentum_rate);
        weights_ih_velocity.add(wih_delta);
        weights_ih.add(weights_ih_velocity);
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
            double target = desiredOutput[i][0];
            double error = target - prediction.get(0);
            totalLoss += error * error; // sum square error
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
}
