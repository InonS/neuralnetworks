package com.github.neuralnetworks.test;

import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.aparapi.internal.kernel.KernelPreferences;
import com.aparapi.internal.kernel.KernelProfile;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.util.Environment;
import org.junit.Test;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.LongSummaryStatistics;
import java.util.StringJoiner;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

import static java.lang.Thread.sleep;
import static java.util.Arrays.stream;
import static java.util.stream.IntStream.range;
import static org.junit.Assert.assertEquals;

/**
 * neuralnetworks
 * Load test GPU with the general feedforward neural networks test from {@link FFNNTest#testWeightedSumBP}.
 * Created by Inon Sharony on 2017-07-11.
 */

public class GPULoadTest {

    private static final PrintStream out = System.out;

    private static final int COOLDOWN_MILLIS = 2000;
    private static final int WARMUP_ITERATIONS = 5;
    private static final int MEASUREMENT_ITERATIONS = 10;

    private static final Environment ENV = Environment.getInstance();
    private static final KernelManager KM = KernelManager.instance();

    /* JMH annotations:
        @Benchmark()
        @BenchmarkMode(Mode.All)
        @Warmup(iterations = 5)
        @Measurement(iterations = 5)
        @Fork(1)
        @OutputTimeUnit(TimeUnit.MILLISECONDS)
    */
    @Test
    public void weightedSumBPLoadTest() throws InterruptedException {
        LongSupplier timeSupplier = getTimeSupplier(TimeUnit.MILLISECONDS);
        stream(EXECUTION_MODE.values()).forEach(executionMode -> {
                    try {
                        sleep(COOLDOWN_MILLIS);
                        out.print("\n" + executionMode + ": ");
                        range(0, WARMUP_ITERATIONS).forEach(x -> testWeightedSumBP(executionMode));
                        LongSummaryStatistics stats = range(0, MEASUREMENT_ITERATIONS).mapToLong(x -> timeit(executionMode, timeSupplier)).summaryStatistics();
                        out.println(stats.toString());
                        out.flush();
                    } catch (InterruptedException ie) {
                        ie.printStackTrace();
                    }
                }
        );
    }

    private LongSupplier getTimeSupplier(TimeUnit outputTimeUnits) {
        switch (outputTimeUnits) {
            case NANOSECONDS:
                return System::nanoTime;
            case MILLISECONDS:
                return System::currentTimeMillis;
            default:
                return null;
        }
    }

    private long timeit(EXECUTION_MODE executionMode, LongSupplier timeSupplier) {
        long start = timeSupplier.getAsLong();
        testWeightedSumBP(executionMode);
        return timeSupplier.getAsLong() - start;
    }

    private void testWeightedSumBP(EXECUTION_MODE executionMode) {
        ENV.setExecutionMode(executionMode);
        interrogateExecutionStrategy();

        Layer il1 = new Layer();
        Layer ol = new Layer();
        Layer il2 = new Layer();

        Tensor weights = TensorFactory.tensor(2, 3, 2);
        FullyConnected c1 = new FullyConnected(ol, il1, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 2, 1}}));
        FullyConnected c2 = new FullyConnected(ol, il2, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 2, 1}}));
        FullyConnected bc = new FullyConnected(new Layer(), ol, 1, 2);

        Matrix cg = c1.getWeights();
        cg.set(1, 0, 0);
        cg.set(2, 1, 0);
        cg.set(3, 2, 0);
        cg.set(4, 0, 1);
        cg.set(5, 1, 1);
        cg.set(6, 2, 1);

        cg = c2.getWeights();
        cg.set(1, 0, 0);
        cg.set(2, 1, 0);
        cg.set(3, 2, 0);
        cg.set(4, 0, 1);
        cg.set(5, 1, 1);
        cg.set(6, 2, 1);

        Matrix bcg = bc.getWeights();
        bcg.set(0.1f, 0, 0);
        bcg.set(0.2f, 1, 0);

        ConnectionCalculatorFullyConnected aws = new AparapiWeightedSumConnectionCalculator();

        List<Connections> connections = new ArrayList<>();
        connections.add(c1);
        NeuralNetworkImpl nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));
        ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, true);

        Matrix i1 = vp.get(il1);
        i1.set(1, 0, 0);
        i1.set(2, 1, 0);
        i1.set(3, 2, 0);
        i1.set(4, 0, 1);
        i1.set(5, 1, 1);
        i1.set(6, 2, 1);

        interrogateKernel(connections, vp, ol);

        aws.calculate(connections, vp, ol);

        // most simple case
        Matrix o = vp.get(ol);
        assertEquals(14, o.get(0, 0), 0);
        assertEquals(32, o.get(0, 1), 0);
        assertEquals(32, o.get(1, 0), 0);
        assertEquals(77, o.get(1, 1), 0);

        // with bias
        connections = new ArrayList<>();
        connections.add(c1);
        connections.add(bc);
        nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));
        vp = TensorFactory.tensorProvider(nn, 2, true);
        i1 = vp.get(il1);
        i1.set(1, 0, 0);
        i1.set(2, 1, 0);
        i1.set(3, 2, 0);
        i1.set(4, 0, 1);
        i1.set(5, 1, 1);
        i1.set(6, 2, 1);

        aws = new AparapiWeightedSumConnectionCalculator();
        aws.calculate(connections, vp, ol);

        o = vp.get(ol);
        assertEquals(14.1, o.get(0, 0), 0.01);
        assertEquals(32.1, o.get(0, 1), 0.01);
        assertEquals(32.2, o.get(1, 0), 0.01);
        assertEquals(77.2, o.get(1, 1), 0.01);

        // combined layers
        connections = new ArrayList<>();
        connections.add(c1);
        connections.add(c2);
        connections.add(bc);
        nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));
        vp = TensorFactory.tensorProvider(nn, 2, true);

        i1 = vp.get(il1);
        i1.set(1, 0, 0);
        i1.set(2, 1, 0);
        i1.set(3, 2, 0);
        i1.set(4, 0, 1);
        i1.set(5, 1, 1);
        i1.set(6, 2, 1);

        Matrix i2 = vp.get(il2);
        i2.set(1, 0, 0);
        i2.set(2, 1, 0);
        i2.set(3, 2, 0);
        i2.set(4, 0, 1);
        i2.set(5, 1, 1);
        i2.set(6, 2, 1);

        aws = new AparapiWeightedSumConnectionCalculator();
        aws.calculate(connections, vp, ol);

        o = vp.get(ol);
        assertEquals(28.1, o.get(0, 0), 0.01);
        assertEquals(64.1, o.get(0, 1), 0.01);
        assertEquals(64.2, o.get(1, 0), 0.01);
        assertEquals(154.2, o.get(1, 1), 0.01);
    }

    private void interrogateExecutionStrategy() {
        String executionStrategyName = ENV.getExecutionStrategy().getClass().getSimpleName();
        out.println("ExecutionStrategy: " + executionStrategyName);
    }

    private void interrogateKernel(List<Connections> connections, ValuesProvider vp, Layer ol) {
        AparapiWeightedSum kernel = new AparapiWeightedSum(connections, vp, ol);

        KernelProfile profile = KM.getProfile(kernel.getClass());
        out.println("devices:");
        profile.getDevices().forEach(this::interrogateDevice);
        out.println("device profiles:");
        profile.getDeviceProfiles().forEach(deviceProfile -> out.println(deviceProfile.toString()));

        KernelPreferences preferences = KM.getPreferences(kernel);
        out.println("preferred devices:");
        preferences.getPreferredDevices(kernel).forEach(this::interrogateDevice);
        out.println("failed devices:");
        preferences.getFailedDevices().forEach(this::interrogateDevice);
    }

    private void interrogateDevice(Device device) {
        long id = device.getDeviceId();
        Device.TYPE type = device.getType();
        String shortDescription = device.getShortDescription();
        out.println(new StringJoiner(",").add(String.valueOf(id)).add(type.name()).add(shortDescription).toString());

    }
}
