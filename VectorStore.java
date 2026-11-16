import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * Step 1: brute-force vector store.
 * Stores points and finds the k closest to a query by checking every point.
 * Always correct, but O(N * D) per search.
 */
public class VectorStore {

    private final int dimensions;
    private final List<Integer> ids = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();

    public VectorStore(int dimensions) {
        this.dimensions = dimensions;
    }

    /** Store a point. */
    public void add(int id, float[] vector) {
        if (vector.length != dimensions) {
            throw new IllegalArgumentException(
                "Expected " + dimensions + " dimensions, got " + vector.length);
        }
        ids.add(id);
        vectors.add(vector);
    }

    public int size() {
        return ids.size();
    }

    /** Ids of the k closest points to the query, closest first. */
    public List<Integer> search(float[] query, int k) {
        // Max-heap on distance: the top is the farthest of the best k so far.
        // Each entry is {distance, index into the lists}.
        PriorityQueue<float[]> heap =
            new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

        for (int i = 0; i < vectors.size(); i++) {
            float dist = squaredDistance(query, vectors.get(i));
            if (heap.size() < k) {
                heap.add(new float[] {dist, i});
            } else if (dist < heap.peek()[0]) {
                heap.poll();
                heap.add(new float[] {dist, i});
            }
        }

        // The heap pops farthest first, so reverse at the end.
        List<Integer> result = new ArrayList<>();
        while (!heap.isEmpty()) {
            result.add(ids.get((int) heap.poll()[1]));
        }
        Collections.reverse(result);
        return result;
    }

    /**
     * Squared Euclidean distance. We skip the sqrt: it does not change
     * which point is closer, and it saves time.
     */
    static float squaredDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    public static void main(String[] args) {
        // Test 1: small case you can verify by hand.
        VectorStore small = new VectorStore(2);
        small.add(1, new float[] {0, 0});
        small.add(2, new float[] {1, 1});
        small.add(3, new float[] {5, 5});
        small.add(4, new float[] {-1, 0});
        small.add(5, new float[] {10, 10});
        // Query (0.9, 0.9): closest is id 2, then id 1, then id 4.
        System.out.println("Small test, expect [2, 1, 4]: "
            + small.search(new float[] {0.9f, 0.9f}, 3));

        // Test 2: 100,000 random points with 128 dimensions.
        int n = 100_000, d = 128, k = 10;
        Random random = new Random(42);
        VectorStore big = new VectorStore(d);
        for (int i = 0; i < n; i++) {
            big.add(i, randomVector(random, d));
        }

        // Warm up so the JIT compiler has optimised the code before we time it.
        for (int i = 0; i < 20; i++) {
            big.search(randomVector(random, d), k);
        }

        int queries = 100;
        long start = System.nanoTime();
        for (int i = 0; i < queries; i++) {
            big.search(randomVector(random, d), k);
        }
        double msPerQuery = (System.nanoTime() - start) / 1e6 / queries;
        System.out.printf("%d points, %d dims, k=%d: %.2f ms per search%n",
            n, d, k, msPerQuery);
    }

    private static float[] randomVector(Random random, int d) {
        float[] v = new float[d];
        for (int i = 0; i < d; i++) {
            v[i] = random.nextFloat();
        }
        return v;
    }
}
