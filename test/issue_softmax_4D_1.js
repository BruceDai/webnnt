describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax float with 4D input tensor example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input_value = [10.63, 18.75, 12.91, 9.46, 7.31, 12.48, 9.55, 14.28];
    let output_expect = [0.0002926, 0.9835261, 0.0028609, 0.0000908, 0.0000106, 0.0018611, 0.0000994, 0.0112587];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 8]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type0_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
