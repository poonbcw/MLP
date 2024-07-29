import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        /////////////////////////////////////////  Question 1  //////////////////////////////////////////////
//        Matrix m = new Matrix("src/FloodDataSet");
//        NeuralNetwork2 nn2 = new NeuralNetwork2(8, new int[]{8, 6, 4}, 1, 0.057, 0.9);
//        // Normalize all input data
//        double[][] normalizedInputData = new double[m.data.length][m.data[0].length];
//        for (int i = 0; i < m.data.length; i++) {
////            normalizedInputData[i] = nn.normalize(m.data[i], Matrix.dataMin, Matrix.dataMax);
//            normalizedInputData[i] = nn2.normalize(m.data[i], 95, 628);
//        }
//
//        // Normalize desired output
//        double[][] normalizedDesiredOutput = new double[m.desiredOutput.length][m.desiredOutput[0].length];
//        for (int i = 0; i < m.desiredOutput.length; i++) {
////            normalizedDesiredOutput[i] = nn.normalize(m.desiredOutput[i], Matrix.desiredOutputMin, Matrix.desiredOutputMax);
//            normalizedDesiredOutput[i] = nn2.normalize(m.desiredOutput[i], 95, 628);
//        }
//
//        // Train the neural network
//        nn2.fit(normalizedInputData, normalizedDesiredOutput, 10000, 10);
//
//        // Example forward pass
//        List<Double> normalizedOutput = nn2.predict(normalizedInputData[150]);
//
//        // Convert normalized output list to an array for denormalization
//        double[] normalizedOutputArray = new double[normalizedOutput.size()];
//        for (int i = 0; i < normalizedOutput.size(); i++) {
//            normalizedOutputArray[i] = normalizedOutput.get(i);
//        }
//
//        // Denormalize the output
////        double[] denormalizedOutput = nn.denormalize(normalizedOutputArray, Matrix.desiredOutputMin, Matrix.desiredOutputMax);
//        double[] denormalizedOutput = nn2.denormalize(normalizedOutputArray, 95, 628);
//
//        // Print the outputs
//        System.out.println("Normalized output: " + Arrays.toString(normalizedOutputArray));
//        System.out.println("Denormalized output: " + Arrays.toString(denormalizedOutput));
//
//        // Compare with the original desired output
//        System.out.println("Original desired output: " + Arrays.toString(m.desiredOutput[150]));

        /////////////////////////////////////////  Question 2  //////////////////////////////////////////////
        Matrix m = new Matrix("src/cross.pat");
        NeuralNetwork2 nn2 = new NeuralNetwork2(2, new int[]{20}, 2, 0.01, 0.9);

        nn2.fit(m.data, m.desiredOutput, 10000, 10);
        List<Double> output = nn2.predict(m.data[2]);
        // Convert normalized output list to an array
        double[] outputArray = new double[output.size()];
        for (int i = 0; i < output.size(); i++) {
            outputArray[i] = output.get(i);
        }

        // Print the outputs
        System.out.println("Output: " + Arrays.toString(nn2.classifySample(outputArray)));
        System.out.println("Original desired output: " + Arrays.toString(m.desiredOutput[2]));
    }
}
