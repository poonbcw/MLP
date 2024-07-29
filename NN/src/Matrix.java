import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class Matrix {
    double[][] data, desiredOutput;
    int numRows, numCols;
    static double dataMin, dataMax, desiredOutputMin, desiredOutputMax;

    // Constructor for biases
    public Matrix(int numRows, int numCols, boolean isBias) {
        data = new double[numRows][numCols];
        this.numRows = numRows;
        this.numCols = numCols;
        if (isBias) {
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    data[i][j] = 1;  // Initialize biases to 1
                }
            }
        } else {
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    data[i][j] = 0;
                }
            }
        }
    }

    // Constructor for weights
    public Matrix(int numRows, int numCols) {
        data = new double[numRows][numCols];
        this.numRows = numRows;
        this.numCols = numCols;
        double lowerBound = -1 / Math.sqrt(numCols);
        double upperBound = 1 / Math.sqrt(numCols);
        Random random = new Random();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                data[i][j] = lowerBound + (upperBound - lowerBound) * random.nextDouble();
            }
        }
    }

    public Matrix(String filePath) {
        if (Objects.equals(filePath, "src/FloodDataSet")) {
            int headerLines = 2; // Number of header lines to skip
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                int lineCount = 0;

                while ((line = br.readLine()) != null) {
                    if (lineCount++ < headerLines) {
                        continue; // Skip header lines
                    }

                    String[] values = line.trim().split("\\s+");
                    numCols = values.length; // Update the number of columns (should be consistent across lines)
                    numRows++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Initialize double[][] data and desiredOutput arrays
            data = new double[numRows][numCols - 1]; // Excluding the last column
            desiredOutput = new double[numRows][numCols - data[0].length]; // Each row has the output values

            // Second pass: Read the data into the arrays
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                int lineCount = 0;
                int rowIndex = 0;

                while ((line = br.readLine()) != null) {
                    if (lineCount++ < headerLines) {
                        continue; // Skip header lines
                    }

                    String[] values = line.trim().split("\\s+");
                    for (int i = 0; i < data[rowIndex].length; i++) {
                        data[rowIndex][i] = Double.parseDouble(values[i]);
                    }
                    for (int i = data[rowIndex].length; i < values.length; i++) {
                        desiredOutput[rowIndex][i - data[rowIndex].length] = Double.parseDouble(values[i]);
                    }

                    rowIndex++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            List<double[]> inputList = new ArrayList<>();
            List<double[]> outputList = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("p")) {
                        // Skip the current line
                        continue;
                    }

                    // Read input line
                    String[] inputValues = line.split("\\s+");
                    double[] input = {Double.parseDouble(inputValues[0]), Double.parseDouble(inputValues[1])};
                    inputList.add(input);

                    // Read output line
                    line = br.readLine(); // Read the next line for desired output
                    String[] outputValues = line.split("\\s+");
                    double[] output = {Double.parseDouble(outputValues[0]), Double.parseDouble(outputValues[1])};
                    outputList.add(output);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            numRows = inputList.size();
            numCols = inputList.get(0).length;

            data = new double[numRows][numCols];
            desiredOutput = new double[numRows][outputList.get(0).length];

            for (int i = 0; i < numRows; i++) {
                data[i] = inputList.get(i);
                desiredOutput[i] = outputList.get(i);
            }
        }

        dataMin = findMin(data);
        dataMax = findMax(data);
        desiredOutputMin = findMin(desiredOutput);
        desiredOutputMax = findMax(desiredOutput);
//        System.out.println(dataMin);
//        System.out.println(dataMax);
//        System.out.println(desiredOutputMin);
//        System.out.println(desiredOutputMax);

    }

    public void add(double scalar) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                this.data[i][j] += scalar;
            }
        }
    }

    public void add(Matrix m) {
        if (numCols != m.numCols || numRows != m.numRows) {
            System.out.println("Shape Mismatch");
            return;
        }
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                this.data[i][j] += m.data[i][j];
            }
        }
    }

    public static Matrix subtract(Matrix a, Matrix b) {
        Matrix temp = new Matrix(a.numRows, a.numCols);
        for (int i = 0; i < a.numRows; i++) {
            for (int j = 0; j < a.numCols; j++) {
                temp.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return temp;
    }

    public static Matrix transpose(Matrix a) {
        Matrix temp = new Matrix(a.numCols, a.numRows);
        for (int i = 0; i < a.numRows; i++) {
            for (int j = 0; j < a.numCols; j++) {
                temp.data[j][i] = a.data[i][j];
            }
        }
        return temp;
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        Matrix temp = new Matrix(a.numRows, b.numCols);
        for (int i = 0; i < temp.numRows; i++) {
            for (int j = 0; j < temp.numCols; j++) {
                double sum = 0;
                for (int k = 0; k < a.numCols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                temp.data[i][j] = sum;
            }
        }
        return temp;
    }

    public void multiply(Matrix a) {
        for (int i = 0; i < a.numRows; i++) {
            for (int j = 0; j < a.numCols; j++) {
                this.data[i][j] *= a.data[i][j];
            }
        }
    }

    public void multiply(double a) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                this.data[i][j] *= a;
            }
        }
    }

    public void sigmoid() {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++)
                this.data[i][j] = 1 / (1 + Math.exp(-this.data[i][j]));
        }
    }

    public Matrix dsigmoid() {
        Matrix temp = new Matrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++)
                temp.data[i][j] = this.data[i][j] * (1 - this.data[i][j]);
        }
        return temp;
    }

    public static Matrix fromArray(double[] x) {
        Matrix temp = new Matrix(x.length, 1);
        for (int i = 0; i < x.length; i++)
            temp.data[i][0] = x[i];
        return temp;
    }

    public List<Double> toArray() {
        List<Double> temp = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                temp.add(data[i][j]);
            }
        }
        return temp;
    }

    public double findMin(double[][] data) {
        double min = Double.MAX_VALUE;
        for (double[] datum : data) {
            for (double v : datum) {
                if (v < min) {
                    min = v;
                }
            }
        }
        return min;
    }

    public double findMax(double[][] data) {
        double max = Double.MIN_VALUE;
        for (double[] datum : data) {
            for (double v : datum) {
                if (v > max) {
                    max = v;
                }
            }
        }
        return max;
    }

    public void print() {
        for (double[] datum : data) {
            for (int j = 0; j < numCols; j++) {
                System.out.print(datum[j] + " ");
            }
            System.out.println();
        }
    }
}
