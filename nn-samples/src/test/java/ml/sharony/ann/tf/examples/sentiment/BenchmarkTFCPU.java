package ml.sharony.ann.tf.examples.sentiment;

import com.aparapi.Kernel;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import org.jfree.base.log.DefaultLog;
import org.jfree.base.log.LogConfiguration;
import org.jfree.util.Log.SimpleMessage;
import org.jfree.util.PrintStreamLogTarget;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.FileNotFoundException;
import java.net.URL;
import java.util.*;
import java.util.stream.Collectors;

import static java.lang.Math.max;
import static java.lang.System.currentTimeMillis;
import static java.util.Objects.isNull;
import static org.jfree.util.Log.debug;
import static org.jfree.util.Log.info;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class BenchmarkTFCPU {

    private static final ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();

    private static final Environment ENV = Environment.getInstance();

    private static final float learningRate = 0.001f;
    private static final float momentum = 0.0f;
    private static final int trainingBatchSize = 100;
    private static final int testBatchSize = 1000;
    private static final int epochs = 1; // 10

    private static String trainImagesPath;
    private static String trainLabelsPath;
    private static String testImagesPath;
    private static String testLabelsPath;

    private static long startMillis;

    static {
        setupLogger();
        try {
            locateDatasetResources();
        } catch (FileNotFoundException fnfe) {
            fnfe.printStackTrace();
        }
    }

    @Parameterized.Parameter()
    public Kernel.EXECUTION_MODE executionMode;

    private static void locateDatasetResources() throws FileNotFoundException {
        trainImagesPath = getResourcePath("train-images.idx3-ubyte").orElseThrow(FileNotFoundException::new);
        trainLabelsPath = getResourcePath("train-labels.idx1-ubyte").orElseThrow(FileNotFoundException::new);
        testImagesPath = getResourcePath("t10k-images.idx3-ubyte").orElseThrow(FileNotFoundException::new);
        testLabelsPath = getResourcePath("t10k-labels.idx1-ubyte").orElseThrow(FileNotFoundException::new);
    }

    private static Optional<String> getResourcePath(String resourceFilename) {
        URL resource = contextClassLoader.getResource(resourceFilename);
        return isNull(resource) ? Optional.empty() : Optional.of(resource.getPath());
    }

    private static void setupLogger() {
        String logLevel = "Debug"; // not {int logLevel = LogTarget.INFO;}
        LogConfiguration.setLogLevel(logLevel);
        DefaultLog log = DefaultLog.getDefaultLog();
        log.init();
        log.addTarget(new PrintStreamLogTarget());
        DefaultLog.installDefaultLog();
    }

    @Parameterized.Parameters
    public static Iterable<Object[]> parameterValuesProvider() {
        return Arrays.stream(Kernel.EXECUTION_MODE.values()).map(mode -> Collections.singletonList(mode).toArray()).collect(Collectors.toList());
    }

    /**
     * {MnistTest#testSigmoidBP}
     */
    @Test
    public void test() {
        info(new SimpleMessage("Starting test with execution mode ", executionMode));
        startMillis = currentTimeMillis();
        ENV.setUseDataSharedMemory(false);
        ENV.setUseWeightsSharedMemory(false);

        debug(new SimpleMessage(relativeMillisAsString(), " importing dataset"));
        MnistInputProvider trainInputProvider = new MnistInputProvider(trainImagesPath, trainLabelsPath);
        trainInputProvider.addInputModifier(new ScalingInputFunction(255));
        MnistInputProvider testInputProvider = new MnistInputProvider(testImagesPath, testLabelsPath);
        testInputProvider.addInputModifier(new ScalingInputFunction(255));

        debug(new SimpleMessage(relativeMillisAsString(), " Training set size (number of samples): ", trainInputProvider.getInputSize()));
        float[] firstInput = trainInputProvider.getNextInput();
        float[] firstTarget = trainInputProvider.getNextTarget();
        trainInputProvider.reset();

        int nInputLayerNodes = firstInput.length;
        int nOutputLayerNodes = firstTarget.length;
        int nHiddenLayerNodes = max(100, nOutputLayerNodes);
        int[] n_nodes_per_layer = {nInputLayerNodes, nHiddenLayerNodes, nHiddenLayerNodes, nHiddenLayerNodes, nOutputLayerNodes};
        String nNodesPerLayerAsString = Arrays.stream(n_nodes_per_layer).mapToObj(String::valueOf).collect(Collectors.joining(" * "));
        debug(new SimpleMessage(relativeMillisAsString(), " Model size ", nNodesPerLayerAsString));

        info(relativeMillisAsString() + " building model");
        AparapiReLU reluCc = new AparapiReLU(); // AparapiBackpropagationFullyConnected
        NeuralNetworkImpl mlp = NNFactory.mlpRelu(n_nodes_per_layer, true, reluCc);

        info(relativeMillisAsString() + " building trainer");
        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), learningRate, momentum, 0f, 0f, 0f, trainingBatchSize, testBatchSize, epochs);

        bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

        ENV.setExecutionMode(executionMode);

        info(relativeMillisAsString() + " training");
        bpt.train();
        info(relativeMillisAsString() + " testing");
        bpt.test();

        info(relativeMillisAsString() + " evaluating");
        float totalNetworkError = bpt.getOutputError().getTotalNetworkError();
        assertEquals(0, totalNetworkError, 0.1);
    }

    private static String relativeMillisAsString() {
        return String.valueOf(currentTimeMillis() - startMillis);
    }

}