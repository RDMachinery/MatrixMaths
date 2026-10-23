
/**
 * Indicates the condition that a <code>Matrix</code> is non invertible: it is not
 * square, is empty, or is singular.
 */
public class NonInvertibleMatrixException extends Exception {
    private static final long serialVersionUID = 1L;

    public NonInvertibleMatrixException(String message) {
        super(message);
    }
}
