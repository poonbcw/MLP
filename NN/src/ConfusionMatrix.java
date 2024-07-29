public class ConfusionMatrix {

    public static void main(String[] args) {
        // Example predicted and actual values
        int[] predicted = {1, 0, 1, 1, 0, 1, 0, 0, 1, 0};
        int[] actual = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0};

        // Initialize counters
        int TP = 0; // True Positives
        int TN = 0; // True Negatives
        int FP = 0; // False Positives
        int FN = 0; // False Negatives

        // Compute confusion matrix
        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == 1 && actual[i] == 1) {
                TP++;
            } else if (predicted[i] == 0 && actual[i] == 0) {
                TN++;
            } else if (predicted[i] == 1 && actual[i] == 0) {
                FP++;
            } else if (predicted[i] == 0 && actual[i] == 1) {
                FN++;
            }
        }

        // Display confusion matrix
        System.out.println("Confusion Matrix:");
        System.out.println("               Actual");
        System.out.println("Predicted    | 1  0");
        System.out.println("           1 | " + TP + "  " + FP);
        System.out.println("           0 | " + FN + "  " + TN);
    }
}
