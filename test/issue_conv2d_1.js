describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv2d v1_2 example-9', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op41_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type5_length = product(type5.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type5);

    model.setOperandValue(op21, new Float32Array([1, 4, 7, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op31, new Float32Array([-200]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type5_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });
})
