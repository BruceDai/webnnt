describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Add 1D-1D example', async function() {
    let model = await nn.createModel(options);
    const TENSOR_SIZE = 1;
    let operandIndex = 0;
    let expect = 11;

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [TENSOR_SIZE]};
    let scalarInt32Type = {type: nn.INT32};

    let fusedActivationFucNone = operandIndex++;
    model.addOperand(scalarInt32Type);
    model.setOperandValue(fusedActivationFucNone, new Int32Array([nn.FUSED_NONE]));

    let tensor0 = operandIndex++;
    model.addOperand(float32TensorType);
    model.setOperandValue(tensor0, new Float32Array([10]));

    let tensor1 = operandIndex++;
    model.addOperand(float32TensorType);
    // model.setOperandValue(tensor0, new Float32Array([1]));

    let intermediateOutput0 = operandIndex++;
    model.addOperand(float32TensorType);

    model.addOperation(nn.ADD, [tensor0, tensor1, fusedActivationFucNone], [intermediateOutput0]);

    model.identifyInputsAndOutputs([tensor1], [intermediateOutput0]);

    await model.finish();

    let compilation = await model.createCompilation();
    // BNNS: 1
    // DNNL:[27144:28447:0211/110852.803541:ERROR:compilation_delegate_dnnl.cc(116)] [DNNL] failed to load DNNL library
    // compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);

    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let outputTensor = new Float32Array(1);
    execution.setInput(0, new Float32Array([1]));
    execution.setOutput(0, outputTensor);
    await execution.startCompute();

    assert.isTrue(almostEqualCTS(expect, outputTensor[0]));

  });
});
